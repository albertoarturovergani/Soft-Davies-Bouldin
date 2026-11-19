"""Extend only capped ten-feature FCM fits, retaining undefined-score diagnostics."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from sdb import fuzzy_c_means
from supplementary_overlap import predict_membership
from supplementary_overlap_ten_features import fcm_scores, make_data, score_means, select_k


def extend(task):
    index, row, config = task
    with threadpool_limits(limits=1):
        x,_ = make_data(int(row['replicate']),row['separation'],config['n_per_class'],n_classes=config['n_classes'])
        c,u,it = fuzzy_c_means(x,int(row['k']),m=2,seed=int(row['seed']),max_iter=10000,tol=config['tol'])
        result = dict(replicate=int(row['replicate']),step=int(row['step']),k=int(row['k']),seed=int(row['seed']),
                      iterations=it,at_limit=it==10000,
                      fixed_point_residual=float(np.max(np.abs(predict_membership(x,c)-u))),**fcm_scores(x,c,u))
    return index,result


def main(directory, workers):
    raw = pd.read_csv(directory/'per_initialization.csv.gz')
    config = json.loads((directory/'provenance.json').read_text())['configuration']
    tasks = [(index,row.to_dict(),config) for index,row in raw[raw.fcm_at_limit].iterrows()]
    changes = []
    if tasks:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for future in as_completed([pool.submit(extend,task) for task in tasks]):
                index,result = future.result()
                for name in ['SDB_fcm','DB_fcm_hard']:
                    result[name+'_primary'] = raw.loc[index,name]
                for name,value in result.items():
                    if name in raw.columns and name not in ['replicate','step','k','seed']:
                        raw.loc[index,name] = value
                changes.append(result)
                if len(changes)%100 == 0 or len(changes)==len(tasks):
                    print(f'C={config["n_classes"]}: extended {len(changes)}/{len(tasks)} capped FCM fits',flush=True)
    columns = ['replicate','step','k','seed','iterations','at_limit','fixed_point_residual',
               'SDB_fcm','SDB_fcm_status','DB_fcm_hard','DB_fcm_hard_status',
               'fcm_min_center_distance','fcm_relative_min_distance','fcm_near_coincident',
               'fcm_hard_clusters','fcm_uniform_deviation','SDB_fcm_primary','DB_fcm_hard_primary']
    runs = pd.DataFrame(changes,columns=columns).sort_values(['replicate','step','k','seed'])
    runs.to_csv(directory/'convergence_sensitivity_runs.csv',index=False,float_format='%.16g')
    original = pd.read_csv(directory/'selected_k.csv')
    extended = select_k(score_means(raw))
    choices = original.merge(extended,on=['replicate','step','metric'],suffixes=('_primary','_extended'))
    choices['changed'] = choices.selected_k_primary != choices.selected_k_extended
    choices.to_csv(directory/'convergence_sensitivity_choices.csv',index=False)
    rates = []
    for (step,metric),group in choices.groupby(['step','metric']):
        rates.append(dict(step=step,metric=metric,
                          primary_correct_rate=float((group.selected_k_primary==config['n_classes']).mean()),
                          extended_correct_rate=float((group.selected_k_extended==config['n_classes']).mean())))
    rates = pd.DataFrame(rates)
    rates['changed'] = rates.primary_correct_rate != rates.extended_correct_rate
    rates.to_csv(directory/'convergence_recovery_rates.csv',index=False,float_format='%.12g')
    select_k(score_means(raw,screened=True)).to_csv(directory/'convergence_distinct_clusters_selected_k.csv',index=False)
    summary = dict(rerun_configurations=len(tasks),extended_max_iter=10000,
                   still_at_limit=int(runs.at_limit.sum()) if len(runs) else 0,
                   maximum_fixed_point_residual=float(runs.fixed_point_residual.max()) if len(runs) else 0.,
                   changed_k_choices=int(choices.changed.sum()),total_k_choices=len(choices),
                   changed_sdb_choices=int(choices.loc[choices.metric=='SDB_fcm','changed'].sum()),
                   changed_recovery_frequencies=int(rates.changed.sum()),
                   undefined_indices_extended={m:int(np.isinf(raw[m]).sum()) for m in ['SDB_fcm','DB_fcm_hard']},
                   note='Primary files retain the original 1,000-iteration protocol. Extended refits use the same seeds and tolerance; degenerate scores remain +inf, never silently omitted.')
    (directory/'convergence_sensitivity.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    args = parser.parse_args()
    main(args.input_dir,args.workers)
