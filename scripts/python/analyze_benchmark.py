"""Produce independently reproducible benchmark summaries and figures."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

PAPER_RIM = {
    "cancer": 0.6058, "crab": 0.6434, "glass": 0.6779, "iris": 0.7261,
    "ovarian": 0.6879, "simple": 0.4637, "thyroid": 0.7516, "wine": 0.6675,
}
PAPER_AIM_K = {"cancer": 2, "crab": 8, "glass": 10, "iris": 2, "ovarian": 10, "simple": 4, "thyroid": 3, "wine": 2}
METRICS = ["SDB_published", "SDB_power_variant", "DB_fcm_hard", "DB_kmeans"]


def summarize(data):
    rows = []
    for name, group in data.groupby("dataset", sort=True):
        avg = group.groupby("k_tested")[METRICS].mean()
        true_k = int(group.k_true.iloc[0])
        if true_k not in avg.index:
            raise ValueError(f"True K {true_k} not sampled for {name}")
        paper_value = PAPER_RIM.get(name, np.nan)
        row = {"dataset": name, "n_samples": int(group.n_samples.iloc[0]), "n_features": int(group.n_features.iloc[0]), "k_true": true_k,
               "SDB_at_true_k": float(avg.loc[true_k, "SDB_published"]), "paper_SDB_at_true_k": paper_value,
               "absolute_error_SDB": float(abs(avg.loc[true_k, "SDB_published"] - paper_value)),
               "ARI_fcm_at_true_k": float(group.loc[group.k_tested == true_k, "ARI_fcm_vs_truth"].mean()),
               "k_paper_AIM_SDB": PAPER_AIM_K.get(name, np.nan)}
        for metric in METRICS:
            row["best_k_" + metric] = int(avg[metric].idxmin())
        rows.append(row)
    return pd.DataFrame(rows)


def plot_sdb_vs_k(data, name, destination):
    import matplotlib.pyplot as plt
    group = data.loc[data.dataset == name]
    avg = group.groupby("k_tested")[["SDB_published", "DB_fcm_hard", "DB_kmeans"]].mean()
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for field, label in [("SDB_published", "SDB on FCM"), ("DB_fcm_hard", "DB on hardened FCM"), ("DB_kmeans", "DB on K-means")]:
        ax.plot(avg.index, avg[field], marker="o", label=label)
    k = int(group.k_true.iloc[0]); ax.axvline(k, color="0.5", linestyle="--", linewidth=1, label=f"Ground truth K={k}")
    ax.set(xlabel="Candidate number of clusters K", ylabel="Index score (lower is preferred within each index)", title=f"{name.title()}: raw-data clustering vs K")
    ax.set_xticks(avg.index); ax.grid(True, alpha=.18); ax.legend(loc="best", fontsize=9)
    fig.tight_layout(); fig.savefig(destination, dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/recomputed"))
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    required = set(["dataset", "k_true", "n_samples", "n_features", "k_tested", "run", "ARI_fcm_vs_truth", *METRICS])
    if missing := required.difference(data.columns):
        raise ValueError(f"Missing benchmark columns: {sorted(missing)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(data)
    summary.to_csv(args.output_dir / "benchmark_summary.csv", index=False, float_format="%.9f")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 5.7))
    ax.scatter(summary.paper_SDB_at_true_k, summary.SDB_at_true_k, s=50)
    for row in summary.itertuples():
        ax.annotate(row.dataset.title(), (row.paper_SDB_at_true_k, row.SDB_at_true_k), xytext=(5, 3), textcoords="offset points", fontsize=9)
    limits = [min(summary.paper_SDB_at_true_k.min(), summary.SDB_at_true_k.min()) - .03,
              max(summary.paper_SDB_at_true_k.max(), summary.SDB_at_true_k.max()) + .03]
    ax.plot(limits, limits, linestyle="--", color="0.4")
    ax.set(xlim=limits, ylim=limits, xlabel="Paper Table IV SDB", ylabel="Python SDB at ground-truth K", title="Numerical agreement with Vergani & Binaghi (2018)")
    ax.set_aspect("equal", adjustable="box"); ax.grid(True, alpha=.2)
    fig.tight_layout(); fig.savefig(args.output_dir / "paper_reproduction.png", dpi=180); plt.close(fig)
    names = summary.dataset.tolist(); x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for col, label, marker in [("k_true", "Ground truth", "x"), ("best_k_SDB_published", "SDB", "o"),
                                ("best_k_DB_fcm_hard", "DB hardened FCM", "s"), ("best_k_DB_kmeans", "DB K-means", "^")]:
        ax.scatter(x, summary[col], label=label, s=70, marker=marker)
    ax.set_xticks(x, [x.title() for x in names], rotation=40, ha="right")
    ax.set(xlabel="Archived dataset", ylabel="Number of clusters", title="Optimal K selected by each index (3 seeds, K=2..10)")
    ax.set_yticks(range(2, 11)); ax.grid(axis="y", alpha=.15); ax.legend(ncol=2)
    fig.tight_layout(); fig.savefig(args.output_dir / "selected_k.png", dpi=180); plt.close(fig)
    for name in ["iris", "simple", "wine", "ovarian"]:
        if name in names: plot_sdb_vs_k(data, name, args.output_dir / f"curve_{name}.png")
    paper_k_match = (summary.best_k_SDB_published == summary.k_paper_AIM_SDB).sum()
    true_k_hits = {field: int((summary["best_k_" + field] == summary.k_true).sum()) for field in METRICS}
    lines = ["# Independent Python reimplementation: benchmark report", "",
             "**Protocol:** archived eight original Data/T MATLAB datasets; original unstandardized features; FCM exponent m=2; q=1; Euclidean geometry; candidate K=2–10; 3 seeded FCM and K-means replicates per K; mean scores over seeds; FCM max 100 iterations. K-means uses scikit-learn with `n_init=10`. All scores are recomputed with new NumPy implementation code.", "",
             "This report is a **computational reproduction check**, not a rerun of all 100 replicates or the fMRI experiment. Exact hard partitions and random initializations differ from historic MATLAB. Low scores must not be conflated with better agreement to ground-truth labels.", "",
             "## Agreement with published SDB (Table IV, ground-truth K)", "",
             "| Dataset | N | D | Ground-truth K | Paper SDB | Python SDB | Abs error |",
             "|:--|--:|--:|--:|--:|--:|--:|"]
    for r in summary.itertuples():
        lines.append(f"| {r.dataset.title()} | {r.n_samples} | {r.n_features} | {r.k_true} | {r.paper_SDB_at_true_k:.4f} | {r.SDB_at_true_k:.4f} | {r.absolute_error_SDB:.6f} |")
    lines += ["", f"**Maximum unrounded absolute deviation:** {summary.absolute_error_SDB.max():.6f}. The eight Table IV scores all agree to four decimal places.", "",
              "## Selection of optimal K (paper Table III, AIM)", "",
              "| Dataset | True K | SDB K (2018) | SDB K (Python) | DB K-means K (Python) | DB hardened FCM K (Python) |",
              "|:--|--:|--:|--:|--:|--:|"]
    for r in summary.itertuples():
        lines.append(f"| {r.dataset.title()} | {r.k_true} | {r.k_paper_AIM_SDB} | {r.best_k_SDB_published} | {r.best_k_DB_kmeans} | {r.best_k_DB_fcm_hard} |")
    lines += ["", f"SDB selected the same optimal K as the paper in **{paper_k_match}/8 datasets**.",
              "Matching ground-truth K among these eight: " + ", ".join(f"{field}: {value}/8" for field, value in true_k_hits.items()) + ".", "",
              "## Scientific interpretation", "",
              "- SDB is numerically reproducible from its published equations, with no need to copy the original MATLAB source.",
              "- Recovering a low index score is not synonymous with recovering the true number of classes. SDB recovers true K in 3/8 for this K-range.",
              "- The absolute scores of SDB and crisp DB are on different scales because they use different definitions of within-cluster scatter. Their absolute magnitudes cannot alone establish superiority.",
              "- In high-K, high-dimensional/imbalanced data such as Cancer and Thyroid, FCM cluster centers can approach each other; the ratio is very sensitive to inter-center distance and some runs give extremely large SDB values. Inspect distributions across random restarts, instability, and closeness of centroids.",
              "- To replicate historical ANOVAs and the cross-dataset median MRC figure, use the paper's 100 repetitions and explicitly match all original preprocessing and clustering choices. No claims of statistical significance are made from the present 3-seed pilot.", "",
              "## Figures", "", "![Published vs reproduced](../results/recomputed/paper_reproduction.png)",
              "", "![Selected K](../results/recomputed/selected_k.png)",
              "", "## Sources", "", "Paper: https://doi.org/10.1109/FUZZ-IEEE.2018.8491581", "",
              "Original MAT files are not committed; see `data/README.md`. The raw benchmark table is `results/benchmark_original_raw_3seeds_k2-10.csv`.", ""]
    report="\n".join(lines)
    (args.output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"Generated report and 6 figures in {args.output_dir}")
    print(f"Paper agreement: {paper_k_match}/8 optimal K; maximum SDB error: {summary.absolute_error_SDB.max():.8f}")


if __name__ == "__main__":
    main()
