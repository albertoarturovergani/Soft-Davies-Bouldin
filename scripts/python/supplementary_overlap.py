"""Supplementary 1D experiment: exact density-overlap steps, not part of the paper.

Run from the repository root. Use --summarize-only to rebuild tables/figures from
the committed scores. No archived/private data are required.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from sdb import davies_bouldin_crisp, fuzzy_c_means, soft_davies_bouldin


INDEXES = ['DB_kmeans', 'DB_fcm_hard', 'SDB_fcm']
ROOT = Path(__file__).resolve().parents[2]


def separation_for_step(step, steps=10):
    """Invert O(d) = integral min(f_left,f_right) = (1-d/2)^2."""
    if steps < 1 or not 0 <= step <= steps:
        raise ValueError('Require 0 <= step <= steps and steps >= 1')
    return float(2 * (1 - np.sqrt(1 - step / steps)))


def density(x, center):
    return np.maximum(1 - np.abs(np.asarray(x) - center), 0)


def generating_centers(separation, n_classes=2):
    return (np.arange(n_classes) - (n_classes - 1)/2) * separation


def posterior_probabilities(x, separation, n_classes=2):
    values = density(np.asarray(x)[:, None], generating_centers(separation, n_classes)[None, :])
    total = values.sum(axis=1, keepdims=True)
    return np.divide(values, total, out=np.full_like(values, 1/n_classes), where=total > 0)


def bayes_accuracy(overlap, n_classes=2):
    """Equal-prior, equally spaced triangular classes with adjacent overlap O."""
    return 1 - (n_classes-1)/n_classes * np.asarray(overlap)


def true_posterior(x, separation):
    """Backwards-compatible binary posterior used in technical note 1."""
    return posterior_probabilities(x, separation, 2)[:, 1]


def make_data(replicate, separation, n_per_class=100, *, test=False, n_classes=2):
    """Independent replicates; paired base samples across the entire sweep."""
    rng = np.random.default_rng(np.random.SeedSequence([20261117, replicate, int(test)]))
    labels = np.repeat(np.arange(n_classes), n_per_class)
    base = rng.triangular(-1., 0., 1., size=labels.size)
    x = base + generating_centers(separation, n_classes)[labels]
    order = rng.permutation(labels.size)
    return x[order, None], labels[order]


def predict_membership(x, centers):
    """FCM m=2 membership at fixed centers, including exact coincidences."""
    distances = np.abs(x - centers[:, 0][None, :])
    zero = distances == 0
    exact = zero.any(axis=1)
    result = np.empty_like(distances)
    result[exact] = zero[exact] / zero[exact].sum(axis=1, keepdims=True)
    ordinary = ~exact
    weights = (distances[ordinary].min(axis=1, keepdims=True) / distances[ordinary]) ** 2
    result[ordinary] = weights / weights.sum(axis=1, keepdims=True)
    return result


def evaluate_best(x, truth, centers, method, separation, n_classes=2):
    order = np.argsort(centers[:, 0])
    centers = centers[order]
    membership = predict_membership(x, centers)
    labels = membership.argmax(axis=1)
    result = {'test_ARI': adjusted_rand_score(truth, labels)}
    if len(centers) == n_classes:
        # Orientation is fixed geometrically, without fitting to the test labels.
        probability = membership if method == 'FCM' else np.eye(n_classes)[labels]
        posterior = posterior_probabilities(x[:, 0], separation, n_classes)
        result.update(
            test_accuracy=float(np.mean(labels == truth)),
            posterior_MSE=float(np.mean((probability - posterior) ** 2)),
            test_Brier=float(np.mean((probability - np.eye(n_classes)[truth]) ** 2)),
            fuzzy_ambiguity=float(np.mean(n_classes/(n_classes-1) * (1-np.sum(probability**2,axis=1)))),
        )
    return result


def run_replicate(config, replicate):
    raw, best = [], []
    n_classes = config.get('n_classes', 2)
    with threadpool_limits(limits=1):
        for step in range(config['steps'] + 1):
            distance = separation_for_step(step, config['steps'])
            x, _ = make_data(replicate, distance, config['n_per_class'], n_classes=n_classes)
            xtest, ytest = make_data(replicate, distance, config['test_per_class'], test=True, n_classes=n_classes)
            for k in range(2, config['max_k'] + 1):
                winners = {}
                for seed in range(config['initializations']):
                    c, u, iterations = fuzzy_c_means(x, k, seed=seed, m=2.,
                                                   max_iter=config['max_iter'], tol=config['tol'])
                    objective = float(np.sum(u ** 2 * (x - c[:, 0][None, :]) ** 2))
                    delta = float(np.max(np.abs(predict_membership(x, c) - u)))
                    model = KMeans(n_clusters=k, n_init=1, random_state=seed,
                                   max_iter=config['max_iter'], tol=config['tol']).fit(x)
                    row = dict(replicate=replicate, step=step, overlap_pct=100 * (1-step/config['steps']),
                               separation=distance, k=k, seed=seed,
                               DB_kmeans=davies_bouldin_crisp(x, model.labels_),
                               DB_fcm_hard=davies_bouldin_crisp(x, u.argmax(axis=1)),
                               SDB_fcm=soft_davies_bouldin(x, c, u),
                               objective_kmeans=float(model.inertia_), objective_fcm=objective,
                               iterations_fcm=iterations, iterations_kmeans=int(model.n_iter_),
                               fcm_fixed_point_residual=delta,
                               fcm_at_limit=iterations == config['max_iter'],
                               kmeans_at_limit=int(model.n_iter_) == config['max_iter'])
                    raw.append(row)
                    for method, obj, centers in [('FCM', objective, c), ('KMeans', model.inertia_, model.cluster_centers_)]:
                        if method not in winners or obj < winners[method][0]:
                            winners[method] = (obj, centers.copy(), seed, row.copy())
                for method, (obj, centers, seed, row) in winners.items():
                    best.append(dict(replicate=replicate, step=step, k=k, method=method, seed=seed,
                                     objective=obj, **{key: row[key] for key in INDEXES},
                                     **evaluate_best(xtest, ytest, centers, method, distance, n_classes)))
    return raw, best


def summarize(directory):
    raw = pd.read_csv(directory / 'per_initialization.csv.gz')
    best = pd.read_csv(directory / 'best_objective_partitions.csv')
    config = json.loads((directory / 'provenance.json').read_text())['configuration']
    n_classes = config.get('n_classes', 2)
    fixed = f'k{n_classes}'
    expected = config['replicates'] * (config['steps']+1) * (config['max_k']-1) * config['initializations']
    assert len(raw) == expected and not raw.duplicated(['replicate', 'step', 'k', 'seed']).any()
    assert np.isfinite(raw[INDEXES].to_numpy()).all()
    grid = raw.groupby(['replicate', 'step', 'k'])[INDEXES].mean().reset_index()
    grid.to_csv(directory / 'mean_scores.csv', index=False, float_format='%.12g')
    selected = []
    for (replicate, step), group in grid.groupby(['replicate', 'step']):
        for metric in INDEXES:
            choice = group.loc[group[metric].idxmin()]
            method = 'KMeans' if metric == 'DB_kmeans' else 'FCM'
            partition = best[(best.replicate == replicate) & (best.step == step) &
                             (best.k == int(choice.k)) & (best.method == method)].iloc[0]
            selected.append(dict(replicate=replicate, step=step, metric=metric,
                                 selected_k=int(choice.k), selected_test_ARI=partition.test_ARI,
                                 selected_mean_score=choice[metric]))
    selected = pd.DataFrame(selected)
    selected.to_csv(directory / 'selected_k.csv', index=False, float_format='%.12g')
    summary = []
    for step in range(config['steps']+1):
        overlap = 1-step/config['steps']
        for metric in INDEXES:
            values = grid.loc[(grid.step == step) & (grid.k == n_classes), metric]
            choices = selected[(selected.step == step) & (selected.metric == metric)]
            summary.append(dict(step=step, overlap_pct=100*overlap,
                                separation=separation_for_step(step, config['steps']), metric=metric,
                                **{f'score_{fixed}_mean':values.mean(), f'score_{fixed}_p10':values.quantile(.1),
                                   f'score_{fixed}_p90':values.quantile(.9), f'selected_{fixed}_rate':(choices.selected_k == n_classes).mean()},
                                selected_test_ARI_mean=choices.selected_test_ARI.mean()))
    pd.DataFrame(summary).to_csv(directory / 'sweep_summary.csv', index=False, float_format='%.12g')
    performance = []
    for (step, method), group in best[best.k == n_classes].groupby(['step', 'method']):
        row = dict(step=step, overlap_pct=100*(1-step/config['steps']), method=method,
                   bayes_accuracy=float(bayes_accuracy(1-step/config['steps'], n_classes)))
        for column in ['test_ARI', 'test_accuracy', 'posterior_MSE', 'test_Brier', 'fuzzy_ambiguity']:
            row.update({column+'_mean': group[column].mean(), column+'_p10': group[column].quantile(.1),
                        column+'_p90': group[column].quantile(.9)})
        performance.append(row)
    pd.DataFrame(performance).to_csv(directory / f'{fixed}_performance.csv', index=False, float_format='%.12g')
    associations = []
    for replicate, group in grid[grid.k == n_classes].groupby('replicate'):
        for metric in INDEXES:
            associations.append(dict(replicate=replicate, metric=metric,
                                     spearman_score_vs_step=spearmanr(group.step, group[metric]).statistic))
    pd.DataFrame(associations).to_csv(directory / 'separation_sensitivity.csv', index=False, float_format='%.12g')
    diagnostics = dict(configurations=len(raw), index_values=int(raw[INDEXES].size),
                       fcm_at_limit=int(raw.fcm_at_limit.sum()), kmeans_at_limit=int(raw.kmeans_at_limit.sum()),
                       max_fcm_fixed_point_residual=float(raw.fcm_fixed_point_residual.max()),
                       finite_index_values=bool(np.isfinite(raw[INDEXES].to_numpy()).all()))
    (directory / 'diagnostics.json').write_text(json.dumps(diagnostics, indent=2)+'\n')
    print(json.dumps(diagnostics), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--classes', dest='n_classes', type=int, choices=[2,3], default=2)
    parser.add_argument('--replicates', type=int, default=20)
    parser.add_argument('--initializations', type=int, default=100)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--n-per-class', type=int, default=100)
    parser.add_argument('--test-per-class', type=int, default=2000)
    parser.add_argument('--max-k', type=int, default=6)
    parser.add_argument('--max-iter', type=int, default=1000)
    parser.add_argument('--tol', type=float, default=1e-7)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = Path('results/supplementary_overlap' + ('_3classes' if args.n_classes == 3 else ''))
    if args.summarize_only:
        summarize(args.output_dir)
        return
    config = {key: value for key, value in vars(args).items() if key not in ['output_dir', 'summarize_only', 'workers']}
    if min(args.replicates, args.initializations, args.steps, args.n_per_class, args.test_per_class, args.workers, args.max_iter) < 1 or not args.n_classes <= args.max_k <= args.n_classes*args.n_per_class or not np.isfinite(args.tol) or args.tol <= 0:
        parser.error('Invalid grid or fitting settings')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    before = time.perf_counter()
    raw, best = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(run_replicate, config, replicate) for replicate in range(args.replicates)]
        for done, job in enumerate(as_completed(jobs), start=1):
            rows, partitions = job.result()
            raw.extend(rows)
            best.extend(partitions)
            print(f'Completed {done}/{args.replicates} independent datasets; {len(raw)} configurations; {time.perf_counter()-before:.1f}s', flush=True)
    raw = pd.DataFrame(raw).sort_values(['replicate', 'step', 'k', 'seed'])
    best = pd.DataFrame(best).sort_values(['replicate', 'step', 'k', 'method'])
    score_path = args.output_dir / 'per_initialization.csv.gz'
    raw.to_csv(score_path, index=False, compression={'method': 'gzip', 'mtime': 0}, float_format='%.16g')
    best.to_csv(args.output_dir / 'best_objective_partitions.csv', index=False, float_format='%.16g')
    provenance = dict(configuration=config, workers=args.workers, python=platform.python_version(),
                      packages={package: version(package) for package in ['numpy','scipy','pandas','scikit-learn','matplotlib','threadpoolctl']},
                      runtime_seconds=time.perf_counter()-before,
                      random_design='SeedSequence([20261117, replicate, test_flag]); fixed base draws across overlap steps',
                      index_definition='Published SDB q=1, p=2, membership_power=1; FCM m=2',
                      kmeans_n_init=1, candidate_k=list(range(2,args.max_k+1)),
                      overlap_definition='Adjacent-pair integral min(f_i, f_(i+1)); equally spaced triangular densities, half-width 1; O(d)=(1-d/2)^2',
                      population_bayes_accuracy='1 - ((C-1)/C)*O; normalized Bayes error equals adjacent overlap O',
                      posterior_error_definition='Mean squared error averaged over observations and class columns; test_Brier is the per-class mean of multiclass Brier components',
                      score_sha256=hashlib.sha256(score_path.read_bytes()).hexdigest(),
                      code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), ROOT/'scripts/python/sdb/fcm.py', ROOT/'scripts/python/sdb/index.py']})
    (args.output_dir/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    summarize(args.output_dir)


if __name__ == '__main__':
    main()
