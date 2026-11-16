"""Re-evaluate the three abstract-level results in Vergani and Binaghi (2018).

Original eight MATLAB Data/T datasets are required in data/original. For fMRI,
place dataset_rest_rio.mat and dataset_task_208_rio.mat in data/original.
Published figures and statistics are not reproduced by copying their values.
"""
import argparse
import csv
from pathlib import Path
import sys
import numpy as np
from scipy.io import loadmat
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
from sdb import soft_davies_bouldin, davies_bouldin_crisp

GROUND_FILES={"Cancer":"cancer.mat","Crab":"crab.mat","Glass":"glass.mat","Iris":"iris.mat","Ovarian":"ovarian.mat","Simple":"simpleclusters.mat","Thyroid":"thyroid.mat","Wine":"wine.mat"}
FMRI_FILES={"Rest":"dataset_rest_rio.mat","Task":"dataset_task_208_rio.mat"}
FIELDS=["dataset","subject","n_samples","n_features","ground_truth_k","k_tested","seed","DB_kmeans","DB_cmeans_hard","SDB_cmeans","FCM_iterations","error"]

def fast_fcm(X,k,m=2.0,seed=0,max_iter=100,tol=1e-6):
    rng=np.random.default_rng(seed)
    n=X.shape[0]
    U=rng.uniform(size=(n,k));U/=U.sum(axis=1,keepdims=True)
    squared=np.einsum('nd,nd->n',X,X)[:,None]
    for iteration in range(1,max_iter+1):
        weights=U**m
        centers=weights.T@X/weights.sum(axis=0)[:,None]
        distances=np.maximum(squared+np.sum(centers**2,axis=1)[None,:]-2*X@centers.T,0)
        zero=distances==0
        zrows=zero.any(axis=1)
        new=np.empty_like(U)
        if zrows.any():new[zrows]=zero[zrows]/zero[zrows].sum(axis=1,keepdims=True)
        if (~zrows).any():
            logw=-np.log(distances[~zrows])/(m-1)
            logw-=logw.max(axis=1,keepdims=True)
            w=np.exp(logw)
            new[~zrows]=w/w.sum(axis=1,keepdims=True)
        delta=np.max(np.abs(new-U))
        U=new
        if delta<tol:break
    weights=U**m
    centers=weights.T@X/weights.sum(axis=0)[:,None]
    return centers,U,iteration

def sources(data_dir,include_fmri):
    for name,fn in GROUND_FILES.items():
        d=loadmat(data_dir/fn,variable_names=['Data','T'])
        X=np.asarray(d['Data'],float)
        T=np.asarray(d['T'])
        if X.ndim!=2 or T.ndim!=2 or T.shape[1]!=X.shape[0] or not np.allclose(T.sum(axis=0),1):raise ValueError(f'Unrecognized dataset {fn}')
        yield name,'',X,int(T.shape[0])
    if include_fmri:
        for name,fn in FMRI_FILES.items():
            d=loadmat(data_dir/fn)
            structs=[(k,v) for k,v in d.items() if not k.startswith('__')]
            if len(structs)!=1:raise ValueError('Ambiguous fMRI file '+fn)
            group=structs[0][1][0,0]
            for sub in group.dtype.names:
                A=np.asarray(group[sub],float)
                if A.ndim!=2 or A.shape[0]!=96:raise ValueError('Unexpected fMRI shape '+str(A.shape))
                yield name+'ROIs',sub,A,None
                yield name+'TOIs',sub,A.T,None

def run_one(X,k,seed):
    center,U,it=fast_fcm(X,k,seed=seed)
    sdb=soft_davies_bouldin(X,center,U)
    dbf=davies_bouldin_crisp(X,np.argmax(U,axis=1))
    labels=KMeans(n_clusters=k,n_init=10,random_state=seed,max_iter=100).fit_predict(X)
    dbk=davies_bouldin_crisp(X,labels)
    return [dbk,dbf,sdb,it]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original-dir',type=Path,default=Path('data/original'))
    p.add_argument('--output',type=Path,default=Path('results/abstract_reproduction/raw_scores.csv'))
    p.add_argument('--ground-runs',type=int,default=100)
    p.add_argument('--fmri-runs',type=int,default=10)
    p.add_argument('--include-fmri',action='store_true')
    p.add_argument('--only',choices=['all','ground','fmri'],default='all')
    p.add_argument('--max-k',type=int,default=10)
    p.add_argument('--min-k',type=int,default=2)
    p.add_argument('--datasets',nargs='*',default=None)
    p.add_argument('--start-at',type=str,default='')
    args=p.parse_args()
    if args.ground_runs<1 or args.fmri_runs<1 or args.max_k<2:raise ValueError('Invalid run counts or candidate K')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    print(f'Running original ground-truth datasets at {args.ground_runs} seeds and fMRI at {args.fmri_runs} seeds',flush=True)
    import time
    with threadpool_limits(limits=1):
        with args.output.open('w',newline='') as file:
            wr=csv.DictWriter(file,fieldnames=FIELDS)
            wr.writeheader()
            current='';completed=0
            for name,subject,X,ktrue in sources(args.original_dir,args.include_fmri or args.only=='fmri'):
                if args.only=='ground' and not ktrue:continue
                if args.only=='fmri' and ktrue:continue
                if args.datasets and name not in args.datasets:continue
                if args.start_at and name<args.start_at:continue
                nrun=args.ground_runs if ktrue else args.fmri_runs
                for k in range(args.min_k,min(args.max_k,X.shape[0]-1)+1):
                    for seed in range(nrun):
                        record=dict(dataset=name,subject=subject,n_samples=X.shape[0],n_features=X.shape[1],ground_truth_k=ktrue or '',k_tested=k,seed=seed)
                        try:
                            dbk,dbf,sdb,it=run_one(X,k,seed)
                            record.update(DB_kmeans=dbk,DB_cmeans_hard=dbf,SDB_cmeans=sdb,FCM_iterations=it,error='')
                        except (ValueError,FloatingPointError) as e:
                            record.update(DB_kmeans='',DB_cmeans_hard='',SDB_cmeans='',FCM_iterations='',error=str(e))
                        wr.writerow(record)
                        completed+=1
                    file.flush()
                    print(f'Progress {name} {subject} k={k} rows={completed}',flush=True)
                file.flush()
                print(f'Done {name:10} {subject:27} rows={completed}',flush=True)
    print(f'Finished {completed} rows -> {args.output}',flush=True)

if __name__=='__main__':main()
