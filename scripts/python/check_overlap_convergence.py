"""Sensitivity check: refit only capped FCM runs with a 10,000-iteration limit."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from supplementary_overlap import INDEXES, make_data, predict_membership
from sdb import davies_bouldin_crisp, fuzzy_c_means, soft_davies_bouldin


def main(directory):
    raw = pd.read_csv(directory/'per_initialization.csv.gz')
    config = json.loads((directory/'provenance.json').read_text())['configuration']
    reruns = []
    with threadpool_limits(limits=1):
        for index, row in raw[raw.fcm_at_limit].iterrows():
            x, _ = make_data(int(row.replicate), row.separation, config['n_per_class'], n_classes=config.get('n_classes',2))
            centers, u, it = fuzzy_c_means(x,int(row.k),m=2,seed=int(row.seed),max_iter=10000,tol=config['tol'])
            scores = {'SDB_fcm':soft_davies_bouldin(x,centers,u),
                      'DB_fcm_hard':davies_bouldin_crisp(x,u.argmax(axis=1))}
            record = {key:row[key] for key in ['replicate','step','k','seed']}
            record.update(iterations=it, at_limit=it==10000,
                          fixed_point_residual=float(np.max(np.abs(predict_membership(x,centers)-u))))
            for metric, score in scores.items():
                record[metric+'_old'] = row[metric]
                record[metric+'_extended'] = score
                raw.loc[index,metric] = score
            reruns.append(record)
    reruns = pd.DataFrame(reruns)
    reruns.to_csv(directory/'convergence_sensitivity_runs.csv',index=False,float_format='%.12g')
    means = raw.groupby(['replicate','step','k'])[INDEXES].mean().reset_index()
    previous = pd.read_csv(directory/'selected_k.csv').set_index(['replicate','step','metric'])
    choices = []
    for (replicate,step), group in means.groupby(['replicate','step']):
        for metric in INDEXES:
            k = int(group.loc[group[metric].idxmin(),'k'])
            before = int(previous.loc[(replicate,step,metric),'selected_k'])
            choices.append(dict(replicate=replicate,step=step,metric=metric,k_primary=before,k_extended=k,changed=before!=k))
    choices = pd.DataFrame(choices)
    choices.to_csv(directory/'convergence_sensitivity_choices.csv',index=False)
    result = dict(rerun_configurations=len(reruns),extended_max_iter=10000,
                  still_at_limit=int(reruns.at_limit.sum()) if len(reruns) else 0,
                  maximum_fixed_point_residual=float(reruns.fixed_point_residual.max()) if len(reruns) else 0.,
                  changed_k_choices=int(choices.changed.sum()),total_k_choices=len(choices),
                  note='Original 1000-iteration results are retained; only capped FCM scores are replaced in this sensitivity check.')
    (directory/'convergence_sensitivity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=Path('results/supplementary_overlap'))
    main(parser.parse_args().input_dir)
