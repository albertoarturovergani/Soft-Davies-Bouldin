"""Evaluate SDB and crisp DB on known-label benchmark datasets."""

import argparse
import csv
from pathlib import Path
import numpy as np
from sdb import fuzzy_c_means, soft_davies_bouldin, davies_bouldin_crisp

DATASETS = ("cancer", "crab", "glass", "iris", "ovarian", "simple", "thyroid", "wine")
ORIGINAL_FILES = {"simple": "simpleclusters.mat", **{name: f"{name}.mat" for name in DATASETS if name != "simple"}}


def load_original(name, directory):
    from scipy.io import loadmat
    filename = Path(directory) / ORIGINAL_FILES[name]
    if not filename.exists():
        raise FileNotFoundError(f"Missing {filename}. See data/README.md")
    record = loadmat(filename, variable_names=["Data", "T"])
    X = np.asarray(record["Data"], dtype=float)
    T = np.asarray(record["T"])
    if T.shape[1] != X.shape[0] or not np.all(T.sum(axis=0) == 1):
        raise ValueError(f"Bad one-hot ground truth in {filename}")
    return X, np.argmax(T, axis=0)


def load_dataset(name, original_dir=None):
    if original_dir:
        return load_original(name, original_dir)
    from sklearn.datasets import load_iris, load_wine
    if name == "iris":
        data = load_iris()
    elif name == "wine":
        data = load_wine()
    else:
        raise ValueError(f"{name} requires --original-dir: see data/README.md")
    return np.asarray(data.data, dtype=float), np.asarray(data.target)


def run_benchmark(names, original_dir, runs, max_k, m, q, scale):
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    for name in names:
        X, truth = load_dataset(name, original_dir)
        if scale == "zscore":
            mu = X.mean(axis=0)
            sd = X.std(axis=0)
            X = (X - mu) / np.where(sd == 0, 1, sd)
        n_true = np.unique(truth).size
        for k in range(2, min(max_k, X.shape[0] - 1) + 1):
            for run in range(runs):
                centers, U, _ = fuzzy_c_means(X, k, m=m, seed=run, max_iter=100)
                hard = np.argmax(U, axis=1)
                hard_db = davies_bouldin_crisp(X, hard)
                sdb_paper = soft_davies_bouldin(X, centers, U, q=q)
                sdb_power = soft_davies_bouldin(X, centers, U, q=q, membership_power=m)
                km = KMeans(n_clusters=k, n_init=10, random_state=run).fit_predict(X)
                db_kmeans = davies_bouldin_crisp(X, km)
                yield {
                    "dataset": name,
                    "n_samples": X.shape[0],
                    "n_features": X.shape[1],
                    "k_true": n_true,
                    "k_tested": k,
                    "run": run,
                    "DB_kmeans": db_kmeans,
                    "DB_fcm_hard": hard_db,
                    "SDB_published": sdb_paper,
                    "SDB_power_variant": sdb_power,
                    "ARI_fcm_vs_truth": adjusted_rand_score(truth, hard),
                    "ARI_kmeans_vs_truth": adjusted_rand_score(truth, km),
                }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["iris", "wine"], help="Dataset names or 'all'")
    parser.add_argument("--original-dir", type=Path, default=None)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-k", type=int, default=5)
    parser.add_argument("--m", type=float, default=2.0, help="FCM fuzzifier")
    parser.add_argument("--q", type=float, default=1.0, help="Within scatter exponent")
    parser.add_argument("--scale", choices=["none", "zscore"], default="none")
    parser.add_argument("--output", type=Path, default=Path("results/benchmark.csv"))
    args = parser.parse_args()
    names = list(DATASETS) if args.datasets == ["all"] else args.datasets
    if not names or any(name not in DATASETS for name in names) or args.runs < 1 or args.max_k < 2:
        parser.error("Invalid datasets, runs, or max-k")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as file:
        writer = None
        count = 0
        for record in run_benchmark(names, args.original_dir, args.runs, args.max_k, args.m, args.q, args.scale):
            if writer is None:
                writer = csv.DictWriter(file, fieldnames=list(record))
                writer.writeheader()
            writer.writerow(record)
            count += 1
    print(f"Saved {count} benchmark rows to {args.output}")
    if args.original_dir is None:
        print("Iris/Wine sourced from scikit-learn, not from the archived 2018 MATLAB files.")
    print("This is a modern reimplementation, not a bitwise replication of the original experiments.")


if __name__ == "__main__":
    main()
