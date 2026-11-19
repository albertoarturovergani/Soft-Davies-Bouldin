"""Diagnose SDB count underestimation without changing the published index.

Read the completed 100-feature studies, reproduce the best native-objective FCM
fit among their 100 starts, and compare formula components on those same fits.
No primary score archive is overwritten. The extra formulae are controls only.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from sdb import fuzzy_c_means, soft_davies_bouldin, davies_bouldin_crisp
from supplementary_overlap import ROOT
from supplementary_overlap_hundred_features import make_data, source_hashes

METRICS = ['SDB', 'SDB_hardened_same_centers', 'coupled_moment_control',
           'conditional_radius_control', 'DB_hardened']


def pair_score(radii, centers):
    distance = np.linalg.norm(centers[:, None] - centers[None, :], axis=2)
    mask = ~np.eye(len(centers), dtype=bool)
    if np.any(distance[mask] <= 1e-12):
        return float('inf'), np.full(len(centers), np.nan), np.full(len(centers), -1)
    np.fill_diagonal(distance, np.inf)
    ratios = (radii[:, None] + radii[None, :]) / distance
    worst = ratios.argmax(axis=1)
    return float(ratios.max(axis=1).mean()), ratios.max(axis=1), worst


def partition_labels(n):
    """Every set partition once, as restricted-growth labels (Bell(n) cases)."""
    def extend(labels):
        if len(labels) == n:
            yield np.array(labels, dtype=int)
        else:
            for value in range(max(labels) + 2):
                yield from extend(labels + [value])
    yield from extend([0])


def noiseless_cases():
    rows = []
    for classes in [4, 5]:
        for geometry in ['line', 'regular_simplex']:
            # Both have minimum inter-class distance 1. This is a deterministic
            # geometry control, not another 100-feature overlap benchmark.
            x = (np.arange(classes, dtype=float)[:, None] if geometry == 'line'
                 else np.eye(classes) / np.sqrt(2))
            for labels in partition_labels(classes):
                k = int(labels.max() + 1)
                if k < 2:
                    continue
                centers = np.array([x[labels == i].mean(axis=0) for i in range(k)])
                try:
                    score = soft_davies_bouldin(x, centers, np.eye(k)[labels])
                except ValueError as error:
                    if str(error) != 'Some cluster centers coincide or are too close':
                        raise
                    score = float('inf')
                rows.append(dict(classes=classes, geometry=geometry, k=k,
                                 labels=','.join(map(str, labels)), SDB=score))
    return pd.DataFrame(rows)


def reproduce(task):
    classes, config, row = task
    rep, k, seed = (int(row[name]) for name in ['replicate', 'k', 'seed'])
    x, truth = make_data(rep, config['separations'][10], config['n_per_class'], n_classes=classes)
    with threadpool_limits(limits=1):
        centers, u, iterations = fuzzy_c_means(x, k, seed=seed, m=2,
                                             max_iter=config['max_iter'], tol=config['tol'])
        radius = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2)
        objective = float(np.sum(u ** 2 * radius ** 2))
        assert np.isclose(objective, row['objective_fcm'], rtol=1e-10, atol=1e-8)
        order = np.argsort(centers.sum(axis=1))
        centers, u, radius = centers[order], u[:, order], radius[:, order]
        mass, global_radius = u.mean(axis=0), radius.mean(axis=0)
        product_of_means = mass * global_radius
        coupled = (u * radius).mean(axis=0)
        conditional = coupled / mass
        hard = u.argmax(axis=1)
        hard_mass = np.bincount(hard, minlength=k) / len(x)
        score, maximum, worst = pair_score(product_of_means, centers)
        expected = row['SDB_fcm']
        assert (np.isinf(score) and np.isinf(expected)) or np.isclose(score, expected, rtol=2e-8, atol=1e-9)
        other = [pair_score(a, centers)[0] for a in [global_radius * hard_mass, coupled, conditional]]
        db = davies_bouldin_crisp(x, hard) if len(np.unique(hard)) > 1 else float('inf')
        result = dict(classes=classes, replicate=rep, k=k, seed=seed, iterations=iterations,
                      objective=objective, ARI=float(adjusted_rand_score(truth, hard)),
                      **dict(zip(METRICS, [score, *other, db])))
        components = [dict(classes=classes, replicate=rep, k=k, cluster=i,
                           center_projection=float(centers[i].sum()/10),
                           mean_membership=float(mass[i]), global_mean_radius=float(global_radius[i]),
                           product_of_means=float(product_of_means[i]), coupled_moment=float(coupled[i]),
                           conditional_radius=float(conditional[i]), worst_neighbor=int(worst[i]),
                           worst_ratio=float(maximum[i])) for i in range(k)]
        memberships = [dict(classes=classes, replicate=rep, k=k, true_class=j, fitted_cluster=i,
                            hard_count=int(((truth == j) & (hard == i)).sum()),
                            mean_membership=float(u[truth == j, i].mean()))
                       for j in range(classes) for i in range(k)]
    return result, components, memberships


def select(values):
    values = values.sort_index()
    values = values[np.isfinite(values)]
    return int(values.idxmin()) if len(values) else 0


def main(output, workers):
    output.mkdir(parents=True, exist_ok=True)
    tasks, aggregations, summaries, archive_hashes = [], [], [], {}
    environment = {p:version(p) for p in ['numpy','scipy','pandas','scikit-learn','matplotlib','threadpoolctl']}
    hashes = source_hashes()
    for classes in [4, 5]:
        directory = ROOT/'results'/f'supplementary_overlap_100features_{classes}classes'
        provenance = json.loads((directory/'provenance.json').read_text())
        assert provenance['code_sha256'] == hashes
        assert provenance['packages'] == environment and provenance['python'] == platform.python_version()
        archive_hashes[str(classes)] = hashlib.sha256((directory/'per_initialization.csv.gz').read_bytes()).hexdigest()
        assert archive_hashes[str(classes)] == provenance['score_sha256']
        raw = pd.read_csv(directory/'per_initialization.csv.gz', float_precision='round_trip')
        raw = raw[raw.step == 10]
        for k, g in raw.groupby('k'):
            summaries.append(dict(classes=classes, k=k, runs=len(g),
                                  mean=float(np.mean(g.SDB_fcm.to_numpy())), median=float(g.SDB_fcm.median()),
                                  minimum=float(g.SDB_fcm.min()), maximum=float(g.SDB_fcm.max()),
                                  nonfinite=int((~np.isfinite(g.SDB_fcm)).sum()),
                                  near_coincident=int(g.fcm_near_coincident.sum()), capped=int(g.fcm_at_limit.sum())))
        for rep, group in raw.groupby('replicate'):
            best = group.loc[group.groupby('k').objective_fcm.idxmin()]
            for rule in ['mean', 'median', 'minimum', 'best_native_objective']:
                if rule == 'best_native_objective':
                    values = best.set_index('k').SDB_fcm
                elif rule == 'mean':
                    values = group.groupby('k').SDB_fcm.agg(lambda a:np.mean(a.to_numpy()))
                else:
                    values = group.groupby('k').SDB_fcm.agg('min' if rule == 'minimum' else rule)
                aggregations.append(dict(classes=classes, replicate=rep, rule=rule, selected_k=select(values)))
            tasks.extend((classes, provenance['configuration'], row.to_dict()) for _, row in best.iterrows())
    results, components, memberships = [], [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result, detail, membership in pool.map(reproduce, tasks):
            results.append(result)
            components.extend(detail)
            memberships.extend(membership)
            if len(results) % 25 == 0:
                print(f'Reproduced {len(results)}/{len(tasks)} selected fits', flush=True)
    frame = pd.DataFrame(results)
    choices = []
    for (classes, rep), group in frame.groupby(['classes','replicate']):
        for metric in METRICS:
            choices.append(dict(classes=classes, replicate=rep, metric=metric,
                                selected_k=select(group.set_index('k')[metric])))
    outputs = dict(endpoint_score_summary=pd.DataFrame(summaries),
                   aggregation_choices=pd.DataFrame(aggregations), best_fcm_fits=frame,
                   formula_components=pd.DataFrame(components), membership_by_true_class=pd.DataFrame(memberships),
                   control_choices=pd.DataFrame(choices), noiseless_partitions=noiseless_cases())
    for name, table in outputs.items():
        table.to_csv(output/(name+'.csv'), index=False, float_format='%.17g')
    manifest = dict(source_archive_sha256=archive_hashes, primary_source_sha256=hashes,
                    diagnostic_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    python=platform.python_version(), packages=environment, reproduced_fits=len(tasks),
                    protocol='Disjoint endpoint only, 20 original datasets for each C=4,5; K=2..6. Reproduce the lowest native FCM objective among the original 100 starts at each K. Verify the original objective and SDB. No selection uses true labels.',
                    controls='Same centers and memberships. Hardened SDB uses hard masses; coupled moment replaces E[r]E[u] with E[ur]; conditional radius replaces it with E[ur]/E[u]. These are diagnostic formula changes, not validated replacement indices. DB_hardened recomputes centers from hard labels.',
                    deterministic='All nontrivial set partitions of 4 and 5 noiseless points, on a line and a regular simplex, minimum pair distance 1. Centers are means, memberships crisp; undefined coincident-center scores are +inf.')
    (output/'provenance.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(pd.DataFrame(choices).groupby(['classes','metric','selected_k']).size().to_string(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results/sdb_undercounting_diagnosis')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    main(args.output_dir, args.workers)
