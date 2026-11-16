"""Summarize independent SDB study and compare to 2018 published results."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

PAPER_AIM={
'Cancer':(0.7628,0.7630,0.6058),'Crab':(0.6113,0.6137,0.5802),'Glass':(0.9167,1.0658,0.6723),'Iris':(0.4048,0.4048,0.5904),
'Ovarian':(0.7487,0.7520,0.5553),'Simple':(0.4428,0.4375,0.4637),'Thyroid':(0.9802,1.0105,0.7516),'Wine':(0.4612,0.4577,0.5524),
'RestROIs':(0.4332,0.4519,0.5563),'RestTOIs':(1.2417,1.2812,1.0565),'TaskROIs':(0.4427,0.4336,0.5618),'TaskTOIs':(1.2432,1.2830,1.0484)}
PAPER_RIM={
'Cancer':(0.7628,0.7630,0.6058),'Crab':(0.6113,0.6137,0.6434),'Glass':(1.0054,1.0752,0.6779),'Iris':(0.6693,0.6696,0.7261),
'Ovarian':(0.7487,0.7520,0.6879),'Simple':(0.4428,0.4375,0.4637),'Thyroid':(1.1370,1.0105,0.7516),'Wine':(0.5412,0.5394,0.6675)}
PAPER_MRC={
2:(0.7564,0.7575,0.6607),3:(0.7410,0.7492,0.7104),4:(0.8596,0.8797,0.6889),5:(0.8572,0.9300,0.6968),
6:(0.8948,1.0860,0.8531),7:(0.8960,1.1997,0.8306),8:(0.9064,1.2995,0.8407),9:(0.9492,1.2069,0.7705),10:(0.9917,1.1550,0.6712)}
ORDER=list(PAPER_AIM)
COLS=['DB_kmeans','DB_cmeans_hard','SDB_cmeans']
GROUND={'Cancer':2,'Crab':2,'Glass':2,'Iris':3,'Ovarian':2,'Simple':4,'Thyroid':3,'Wine':3}
FILE_GROUND=['ground_100_runs.csv','ground_crab_glass_iris_ovarian_100.csv','ground_glass_100.csv','ground_iris_ovarian_wine_100.csv',
'ground_ovarian_k10_100.csv','ground_simple_100.csv','ground_thyroid_k2-5_100.csv','ground_thyroid_k5-6_100.csv',
'ground_thyroid_k6_100.csv','ground_thyroid_k7-8_100.csv','ground_thyroid_k8_100.csv','ground_thyroid_k9_100.csv','ground_thyroid_k10_100.csv','ground_wine_100.csv']
FILE_FMRI=['fmri_restrois_10seeds.csv','fmri_resttois_10seeds.csv','fmri_taskrois_10seeds.csv','fmri_tasktois_10seeds.csv']

def load_scores(directory,input_files=None,ground_runs=100,fmri_runs=10):
    filenames=input_files or [directory/f for f in FILE_GROUND+FILE_FMRI]
    data=pd.concat([pd.read_csv(f) for f in filenames],ignore_index=True)
    data=data.drop_duplicates(['dataset','subject','k_tested','seed'],keep='last')
    if data.error.notna().sum():print('Warning: invalid cases',data.error.value_counts().head())
    expected={**{k:ground_runs*9 for k in GROUND},**{'RestROIs':19*fmri_runs*9,'RestTOIs':19*fmri_runs*9,'TaskROIs':15*fmri_runs*9,'TaskTOIs':15*fmri_runs*9}}
    for k,n in expected.items():
        subset=data[data.dataset==k]
        if len(subset)!=n:raise ValueError(f'{k} has {len(subset)} rows (expected {n})')
        if sorted(subset.k_tested.unique())!=list(range(2,11)):raise ValueError(f'{k} incomplete K range')
        if subset[COLS].isna().any().any():raise ValueError(f'{k} has undefined index values')
    return data

def aggregate(data,stat='mean',ground_runs=100,fmri_runs=10):
    rows=[]
    for name in ORDER:
        group=data.loc[data.dataset==name]
        subject_level=group.groupby(['subject','k_tested'],dropna=False)[COLS].mean().reset_index()
        f=(subject_level.groupby('k_tested')[COLS].mean() if stat=='mean' else subject_level.groupby('k_tested')[COLS].median())
        for k,v in f.iterrows():
            rows.append(dict(dataset=name,k=int(k),n_subjects=group.subject.nunique() if name not in GROUND else 1,replicates=ground_runs if name in GROUND else fmri_runs,
                DB_kmeans=float(v['DB_kmeans']),DB_cmeans_hard=float(v['DB_cmeans_hard']),SDB_cmeans=float(v['SDB_cmeans'])))
    return pd.DataFrame(rows)

def report(agg):
    f=agg.set_index(['dataset','k'])
    aim=[]
    for name in ORDER:
        df=f.loc[name]
        record=dict(dataset=name,true_k=GROUND.get(name,np.nan))
        for field in COLS:
            record[field]=df[field].min();record['best_k_'+field]=int(df[field].idxmin())
        record['SDB_below_both']=bool(record['SDB_cmeans']<min(record['DB_kmeans'],record['DB_cmeans_hard']))
        aim.append(record)
    aim=pd.DataFrame(aim)
    rim=[]
    for name,k in GROUND.items():
        row=dict(dataset=name,k=k)
        for field in COLS:row[field]=float(f.loc[(name,k),field])
        row['SDB_below_both']=row['SDB_cmeans']<min(row['DB_kmeans'],row['DB_cmeans_hard'])
        rim.append(row)
    rim=pd.DataFrame(rim)
    mrc=agg.groupby('k')[COLS].median()
    mrc['SDB_below_both']=mrc['SDB_cmeans']<mrc[['DB_kmeans','DB_cmeans_hard']].min(axis=1)
    return aim,rim,mrc.reset_index()

def compare_stat(aim,rim,mrc,verbose=True):
    n_aim=int(aim.SDB_below_both.sum());n_rim=int(rim.SDB_below_both.sum());n_mrc=int(mrc.SDB_below_both.sum())
    if verbose:
        print('\n-- Independent numerical experiment --')
        print('AIM SDB lower:',n_aim,'/ 12; published 7/12')
        print('RIM SDB lower:',n_rim,'/ 8; published 4/8')
        print('MRC SDB lower:',n_mrc,'/ 9 K=2..10; published 9/9')
        print('\nAIM summary\n',aim.round(4).to_string(index=False))
        print('\nRIM summary\n',rim.round(4).to_string(index=False))
        print('\nMRC summary\n',mrc.round(4).to_string(index=False))
        print('\nReference compared to independent AIM')
        for name,row in aim.set_index('dataset').iterrows():
            pv=PAPER_AIM[name];our=[row[c] for c in COLS]
            print(f'{name:10}', 'published',np.round(pv,4), 'computed',np.round(our,4))
    return n_aim,n_rim,n_mrc

def two_way_anova(data,row_key):
    from scipy.stats import f
    table=data.set_index(row_key)[COLS].to_numpy(dtype=float)
    grand=table.mean()
    row=table.mean(axis=1,keepdims=True)
    col=table.mean(axis=0,keepdims=True)
    n,m=table.shape
    ss_method=n*np.square(col-grand).sum()
    ss_row=m*np.square(row-grand).sum()
    ss_residual=np.square(table-row-col+grand).sum()
    df_method=m-1
    df_row=n-1
    df_residual=df_row*df_method
    ms_residual=ss_residual/df_residual
    F_method=ss_method/df_method/ms_residual
    F_row=ss_row/df_row/ms_residual
    return dict(test=row_key,n_groups=n,F_method=F_method,p_method=float(f.sf(F_method,df_method,df_residual)),F_group=F_row,p_group=float(f.sf(F_row,df_row,df_residual)))

def main():
    pa=argparse.ArgumentParser(description=__doc__)
    pa.add_argument('--input-dir',type=Path,default=Path('results/abstract_reproduction'))
    pa.add_argument('--output-dir',type=Path,default=Path('results/abstract_reproduction'))
    pa.add_argument('--fmri-stat',choices=['mean','median'],default='mean')
    pa.add_argument('--input-files',type=Path,nargs='*',default=None)
    pa.add_argument('--ground-runs',type=int,default=100)
    pa.add_argument('--fmri-runs',type=int,default=10)
    args=pa.parse_args()
    args.output_dir.mkdir(exist_ok=True,parents=True)
    data=load_scores(args.input_dir,args.input_files,args.ground_runs,args.fmri_runs)
    agg=aggregate(data,args.fmri_stat,args.ground_runs,args.fmri_runs)
    aim,rim,mrc=report(agg)
    compare_stat(aim,rim,mrc)
    mrc_zero=pd.concat([pd.DataFrame({'k':[1],**{key:[0.0] for key in COLS}}),mrc],ignore_index=True)
    anova=pd.DataFrame([{'analysis':'AIM',**two_way_anova(aim,'dataset')},{'analysis':'RIM',**two_way_anova(rim,'dataset')},{'analysis':'MRC including historical K=1 placeholder',**two_way_anova(mrc_zero,'k')}])
    anova.to_csv(args.output_dir/'ANOVA_summary.csv',index=False,float_format='%.9g')
    print('Two-way additive ANOVA (descriptive, for comparability):\n',anova.to_string(index=False))
    for name,frame in [('aggregated_scores.csv',agg),('AIM_summary.csv',aim),('RIM_summary.csv',rim),('MRC_summary.csv',mrc)]:frame.to_csv(args.output_dir/name,index=False,float_format='%.8g')
    print('Saved independent results to',args.output_dir)

if __name__=='__main__':main()
