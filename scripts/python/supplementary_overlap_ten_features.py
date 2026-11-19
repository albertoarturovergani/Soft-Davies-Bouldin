"""Ten-feature supplementary sweeps with resumable dataset/step checkpoints.

Published SDB and FCM implementations are unchanged. Undefined index evaluations
are recorded as +inf; a candidate's arithmetic mean includes all 100 starts.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from sdb import davies_bouldin_crisp, fuzzy_c_means, soft_davies_bouldin
from supplementary_overlap import (INDEXES, ROOT, bayes_accuracy, evaluate_best,
                                   generating_centers, predict_membership, squared_distances)
from ten_feature_geometry import calibrate

COLLAPSE_THRESHOLD = 1e-6


def make_data(replicate, separation, n_per_class=100, *, test=False, n_classes=2):
    rng = np.random.default_rng(np.random.SeedSequence([20261117, replicate, int(test)]))
    labels = np.repeat(np.arange(n_classes), n_per_class)
    offsets = generating_centers(separation, n_classes)[labels]
    first = rng.triangular(-1., 0., 1., size=labels.size)+offsets
    order = rng.permutation(labels.size)
    columns = [first]
    for feature in range(2, 11):
        stream = np.random.default_rng(np.random.SeedSequence([20261117, replicate, int(test), feature]))
        columns.append(stream.triangular(-1., 0., 1., size=labels.size)+offsets)
    return np.column_stack(columns)[order], labels[order]


def score_or_infinity(function, *args):
    try:
        value = function(*args)
    except ValueError as error:
        # Only mathematical degeneracy is expected; input/implementation errors
        # must stop the run instead of being hidden as missing scores.
        if str(error) not in ['Some cluster centers coincide or are too close',
                              'Some crisp cluster centroids coincide or are too close',
                              'Classic DB requires at least two nonempty clusters']:
            raise
        return float('inf'), str(error)
    if not np.isfinite(value):
        raise FloatingPointError('Unexpected nonfinite score without degeneracy exception')
    return value, 'ok'


def fcm_scores(x, c, u):
    pair = np.sqrt(squared_distances(c, c))
    pair[np.eye(len(c), dtype=bool)] = np.inf
    minimum = float(pair.min())
    scale = float(np.sqrt(np.mean(np.sum((x-x.mean(axis=0))**2, axis=1))))
    result = dict(fcm_min_center_distance=minimum, fcm_relative_min_distance=minimum/scale,
                  fcm_near_coincident=minimum/scale < COLLAPSE_THRESHOLD,
                  fcm_hard_clusters=int(np.unique(u.argmax(axis=1)).size),
                  fcm_uniform_deviation=float(np.max(np.abs(u-1/u.shape[1]))))
    for metric, fun, args in [('SDB_fcm', soft_davies_bouldin, (x, c, u)),
                              ('DB_fcm_hard', davies_bouldin_crisp, (x, u.argmax(axis=1)))]:
        result[metric], result[metric+'_status'] = score_or_infinity(fun, *args)
    return result


def run_cell(task):
    config, replicate, step = task
    d, classes = config['separations'][step], config['n_classes']
    x, _ = make_data(replicate, d, config['n_per_class'], n_classes=classes)
    xt, yt = make_data(replicate, d, config['test_per_class'], test=True, n_classes=classes)
    raw, best = [], []
    with threadpool_limits(limits=1):
        for k in range(2, config['max_k']+1):
            winners = {}
            for seed in range(config['initializations']):
                c, u, iterations = fuzzy_c_means(x, k, seed=seed, m=2., max_iter=config['max_iter'], tol=config['tol'])
                objective = float(np.sum(u**2*squared_distances(x, c)))
                model = KMeans(n_clusters=k, n_init=1, random_state=seed,
                               max_iter=config['max_iter'], tol=config['tol']).fit(x)
                db, status = score_or_infinity(davies_bouldin_crisp, x, model.labels_)
                row = dict(replicate=replicate, step=step, overlap_pct=100*(1-step/10), separation=d,
                           k=k, seed=seed, DB_kmeans=db, DB_kmeans_status=status,
                           objective_kmeans=float(model.inertia_), objective_fcm=objective,
                           iterations_fcm=iterations, iterations_kmeans=int(model.n_iter_),
                           fcm_at_limit=iterations == config['max_iter'],
                           kmeans_at_limit=int(model.n_iter_) == config['max_iter'],
                           fcm_fixed_point_residual=float(np.max(np.abs(predict_membership(x, c)-u))),
                           **fcm_scores(x, c, u))
                raw.append(row)
                for method, obj, centers, labels in [('FCM', objective, c, u.argmax(axis=1)),
                                                     ('KMeans', model.inertia_, model.cluster_centers_, model.labels_)]:
                    if method not in winners or obj < winners[method][0]:
                        winners[method] = (obj, centers.copy(), seed, int(np.unique(labels).size))
            for method, (obj, centers, seed, occupied) in winners.items():
                best.append(dict(replicate=replicate, step=step, k=k, method=method, seed=seed,
                                 objective=obj, occupied_training_clusters=occupied,
                                 **evaluate_best(xt, yt, centers, method, d, classes, 10)))
    return replicate, step, raw, best


def score_means(raw, *, screened=False):
    frame = raw.copy()
    if screened:
        frame.loc[frame.fcm_near_coincident, 'SDB_fcm'] = np.inf
        frame.loc[frame.fcm_near_coincident | (frame.fcm_hard_clusters != frame.k), 'DB_fcm_hard'] = np.inf
    # np.mean retains +inf; do not silently drop undefined initializations.
    return frame.groupby(['replicate', 'step', 'k'])[INDEXES].agg(lambda x: np.mean(x.to_numpy())).reset_index()


def select_k(means):
    rows = []
    for (replicate, step), group in means.groupby(['replicate', 'step'], sort=True):
        group = group.sort_values('k')
        for metric in INDEXES:
            eligible = group[np.isfinite(group[metric])]
            chosen = int(eligible.loc[eligible[metric].idxmin(), 'k']) if len(eligible) else 0
            rows.append(dict(replicate=replicate, step=step, metric=metric, selected_k=chosen,
                             eligible_candidates=len(eligible)))
    return pd.DataFrame(rows)


def summarize(directory):
    config = json.loads((directory/'provenance.json').read_text())['configuration']
    classes = config['n_classes']
    raw = pd.read_csv(directory/'per_initialization.csv.gz')
    best = pd.read_csv(directory/'best_objective_partitions.csv')
    expected = config['replicates']*11*(config['max_k']-1)*config['initializations']
    assert len(raw) == expected and not raw.duplicated(['replicate','step','k','seed']).any()
    assert not raw[INDEXES].isna().any().any()
    means = score_means(raw)
    means.to_csv(directory/'mean_scores.csv', index=False, float_format='%.16g')
    selected = select_k(means)
    selected.to_csv(directory/'selected_k.csv', index=False)
    screened = select_k(score_means(raw, screened=True))
    screened.to_csv(directory/'distinct_clusters_selected_k.csv', index=False)
    summary = []
    for step in range(11):
        for metric in INDEXES:
            a = selected[(selected.step == step) & (selected.metric == metric)]
            b = screened[(screened.step == step) & (screened.metric == metric)]
            summary.append(dict(step=step, overlap_pct=100*(1-step/10), metric=metric,
                                **{f'selected_k{classes}_rate': float((a.selected_k == classes).mean())},
                                no_eligible_candidate_rate=float((a.selected_k == 0).mean()),
                                distinct_clusters_correct_rate=float((b.selected_k == classes).mean()),
                                distinct_clusters_no_candidate_rate=float((b.selected_k == 0).mean())))
    pd.DataFrame(summary).to_csv(directory/'sweep_summary.csv', index=False, float_format='%.12g')
    performance = []
    for (step, method), group in best[best.k == classes].groupby(['step', 'method']):
        row = dict(step=step, overlap_pct=100*(1-step/10), method=method,
                   bayes_accuracy=float(bayes_accuracy(1-step/10, classes)))
        for column in ['test_ARI','test_accuracy','posterior_MSE','test_Brier','fuzzy_ambiguity']:
            row[column+'_mean'] = group[column].mean()
        performance.append(row)
    pd.DataFrame(performance).to_csv(directory/f'k{classes}_performance.csv', index=False, float_format='%.12g')
    diagnostics = dict(configurations=len(raw), index_values=int(raw[INDEXES].size),
                       undefined_indices={m:int(np.isinf(raw[m]).sum()) for m in INDEXES},
                       fcm_at_limit=int(raw.fcm_at_limit.sum()), kmeans_at_limit=int(raw.kmeans_at_limit.sum()),
                       near_coincident_fcm_fits=int(raw.fcm_near_coincident.sum()),
                       fcm_hard_cluster_count_below_requested=int((raw.fcm_hard_clusters < raw.k).sum()),
                       max_fcm_fixed_point_residual=float(raw.fcm_fixed_point_residual.max()),
                       collapse_relative_threshold=COLLAPSE_THRESHOLD)
    (directory/'diagnostics.json').write_text(json.dumps(diagnostics, indent=2)+'\n')
    pd.DataFrame([dict(step=step, configurations=len(g), near_coincident_fcm_fits=int(g.fcm_near_coincident.sum()),
                      hard_cluster_count_below_requested=int((g.fcm_hard_clusters < g.k).sum()),
                      **{m+'_undefined':int(np.isinf(g[m]).sum()) for m in INDEXES})
                  for step,g in raw.groupby('step')]).to_csv(directory/'degeneracy_by_step.csv', index=False)
    print(json.dumps(diagnostics), flush=True)


def source_hashes():
    names = ['supplementary_overlap_ten_features.py','ten_feature_geometry.py','supplementary_overlap.py',
             'overlap_geometry.py','sdb/fcm.py','sdb/index.py']
    return {name:hashlib.sha256((ROOT/'scripts/python'/name).read_bytes()).hexdigest() for name in names}


def checkpoint_manifest(directory, config):
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {'configuration':config, 'code_sha256':source_hashes()}
    path = directory/'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Checkpoint configuration/source mismatch; use a separate checkpoint directory')
    if not path.exists():
        path.write_text(json.dumps(manifest, indent=2)+'\n')


def run_study(config, directory, checkpoints, workers):
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_manifest(checkpoints, config)
    tasks, completed = [], []
    for rep in range(config['replicates']):
        for step in range(11):
            path = checkpoints/f'replicate_{rep:02}_step_{step:02}.json'
            if path.exists():
                completed.append(json.loads(path.read_text()))
            else:
                tasks.append((config, rep, step))
    before = time.perf_counter()
    print(f'C={config["n_classes"]}, D=10: restored {len(completed)} cells, {len(tasks)} pending', flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(run_cell, task) for task in tasks]):
            rep, step, raw, best = future.result()
            payload = dict(replicate=rep, step=step, raw=raw, best=best)
            path = checkpoints/f'replicate_{rep:02}_step_{step:02}.json'
            temp = path.with_suffix('.tmp')
            temp.write_text(json.dumps(payload)+'\n')
            temp.replace(path)
            completed.append(payload)
            if len(completed)%10 == 0 or len(completed) == config['replicates']*11:
                print(f'C={config["n_classes"]}, D=10: saved {len(completed)}/{config["replicates"]*11} cells; {time.perf_counter()-before:.1f}s', flush=True)
    raw = pd.DataFrame([row for cell in completed for row in cell['raw']]).sort_values(['replicate','step','k','seed'])
    best = pd.DataFrame([row for cell in completed for row in cell['best']]).sort_values(['replicate','step','k','method'])
    score_path = directory/'per_initialization.csv.gz'
    raw.to_csv(score_path,index=False,compression={'method':'gzip','mtime':0},float_format='%.16g')
    best.to_csv(directory/'best_objective_partitions.csv',index=False,float_format='%.16g')
    provenance = dict(configuration=config, workers=workers, python=platform.python_version(),
                      packages={p:version(p) for p in ['numpy','scipy','pandas','scikit-learn','matplotlib','threadpoolctl']},
                      current_invocation_seconds=time.perf_counter()-before,
                      restored_cells=config['replicates']*11-len(tasks), code_sha256=source_hashes(),
                      score_sha256=hashlib.sha256(score_path.read_bytes()).hexdigest(),
                      geometry='Ten informative features; independent Triangular(-1,0,1) noise within class; equal priors; equally spaced centers on (1,...,1).',
                      random_design='Feature 1/order: SeedSequence([20261117,replicate,test_flag]); feature r>=2: independent SeedSequence([20261117,replicate,test_flag,r]); fixed base draws across steps.',
                      index_definition='Published SDB q=1, p=2, membership_power=1; FCM m=2; unchanged implementations.',
                      undefined_score_policy='Mathematical degeneracy returns +inf with a reason; arithmetic mean includes every initialization. Only finite mean candidates are eligible. No eligible candidate is selected_k=0, counted as failure.',
                      primary_selection='Minimize the mean of all starts over requested K=2..6; ties favor smaller K. DB/hardened FCM can use fewer occupied labels than requested; occupancy is recorded.',
                      distinct_cluster_sensitivity='Same means, additionally reject SDB candidates if any start has center separation/data RMS < 1e-6; reject hardened FCM candidates on that condition or any start having fewer occupied labels than requested. This is a diagnostic, not a change to the published index.',
                      overlap_definition='Numerically calibrated adjacent-pair integral min(f_i,f_(i+1)) in R^10; nominal 100%,90%,...,0%. See calibration.json for independent QMC checks.')
    (directory/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    summarize(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--classes', type=int, choices=[2,3,4,5], nargs='+', default=[2,3,4,5])
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--replicates', type=int, default=20)
    parser.add_argument('--initializations', type=int, default=100)
    parser.add_argument('--output-root', type=Path, default=Path('results'))
    parser.add_argument('--checkpoint-root', type=Path, required=True)
    args = parser.parse_args()
    if min(args.workers,args.replicates,args.initializations) < 1:
        parser.error('Counts must be positive')
    calibration_dir = args.output_root/'supplementary_overlap_10features_calibration'
    calibration = calibrate(calibration_dir)
    print('Joint overlap calibration verified: '+str(calibration.reference_absolute_error.max()), flush=True)
    for classes in args.classes:
        directory = args.output_root/f'supplementary_overlap_10features_{classes}classes'
        config = dict(n_classes=classes, n_features=10, replicates=args.replicates,
                      initializations=args.initializations, steps=10, n_per_class=100,
                      test_per_class=2000, max_k=6, max_iter=1000, tol=1e-7,
                      separations=calibration.per_feature_shift.tolist())
        run_study(config,directory,args.checkpoint_root/f'{classes}classes',args.workers)
        for name in ['overlap_calibration.csv','calibration.json']:
            (directory/name).write_bytes((calibration_dir/name).read_bytes())
        print(f'PRIMARY COMPLETE: C={classes}, D=10',flush=True)


if __name__ == '__main__':
    main()
