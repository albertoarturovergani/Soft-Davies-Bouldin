"""Create a six-panel Figure-1-style plot from independently computed AIM/RIM/MRC CSVs."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

KEYS=['DB_kmeans','DB_cmeans_hard','SDB_cmeans']
LABELS=['DB k-means','DB c-means*','SDB c-means']
STYLES=[dict(color='#141414',marker='+',linestyle=':',linewidth=1.4,markersize=7),dict(color='#444444',marker='*',linestyle=':',linewidth=1.2,markersize=8),dict(color='#b02b43',marker='D',linestyle='-',linewidth=1.55,markersize=5,markerfacecolor='white')]
NAMES=['Cancer','Crab','Glass','Iris','Ovarian','Simple','Thyroid','Wine','RestROIs','RestTOIs','TaskROIs','TaskTOIs']

def draw(directory,output):
    a=pd.read_csv(directory/'AIM_summary.csv').set_index('dataset').loc[NAMES]
    r=pd.read_csv(directory/'RIM_summary.csv').set_index('dataset').loc[NAMES[:8]]
    m=pd.read_csv(directory/'MRC_summary.csv').set_index('k').sort_index()
    agg=pd.read_csv(directory/'aggregated_scores.csv')
    ground_runs=int(agg.loc[agg.dataset=='Iris','replicates'].iloc[0])
    fmri_runs=int(agg.loc[agg.dataset=='RestROIs','replicates'].iloc[0])
    fig=plt.figure(figsize=(18,8.8),facecolor='white')
    gs=fig.add_gridspec(2,3,top=.87,bottom=.13,left=.055,right=.975,hspace=.75,wspace=.28,height_ratios=[1.55,.92])
    blocks=[('Absolute indexes minimization (AIM)',a,np.arange(len(a)),a.index.tolist()),('Relative indexes minimization (RIM)',r,np.arange(len(r)),r.index.tolist()),('Measures in relation to clusters (MRC)',m,m.index.to_numpy(),m.index.tolist())]
    for idx,(name,values,x,ticklabels) in enumerate(blocks):
        ax=fig.add_subplot(gs[0,idx])
        for key,label,sty in zip(KEYS,LABELS,STYLES):ax.plot(x,values[key].to_numpy(),label=label,**sty)
        ax.set_title(name,fontsize=13,fontweight='bold',pad=10)
        ax.set_ylabel('Index value' if idx==0 else '')
        ax.set_ylim(0,1.5)
        ax.set_yticks(np.arange(0,1.51,.25))
        ax.grid(axis='both',alpha=.20)
        if idx<2:
            ax.set_xticks(x);ax.set_xticklabels(ticklabels,rotation=47,ha='right',fontsize=9)
        else:
            ax.set_xlim(1.5,10.5);ax.set_xticks(np.arange(2,11));ax.set_xlabel('Number of clusters K')
        ax.legend(fontsize=9,loc='upper left',framealpha=.92)
        bx=fig.add_subplot(gs[1,idx])
        bx.boxplot([values[c].to_numpy() for c in KEYS],widths=.48,showfliers=True,patch_artist=False,
            boxprops=dict(color='#64748b'),whiskerprops=dict(color='#64748b'),capprops=dict(color='#64748b'),
            medianprops=dict(color='#b02b43',linewidth=1.5),flierprops=dict(marker='+',markersize=6,markeredgecolor='#b02b43'))
        bx.set_xticks([1,2,3]);bx.set_xticklabels(LABELS,rotation=25,ha='right',fontsize=9)
        bx.set_ylim(0,1.5);bx.set_yticks(np.arange(0,1.51,.25));bx.set_title(f'Measures boxplot ({["AIM","RIM","MRC"][idx]})',fontsize=12,fontweight='bold')
        bx.grid(axis='y',alpha=.20)
    fig.suptitle('Independent numerical reproduction of Vergani & Binaghi (2018)',fontsize=19,fontweight='bold',y=.985)
    fig.text(.5,.927,f'Eight original labeled benchmarks ({ground_runs} seeds/K) · 19 REST and 15 TASK subjects, ROI/TOI ({fmri_runs} seeds/K) · K = 2–10',ha='center',fontsize=11,color='#475569')
    fig.text(.5,.035,'Within-index comparisons only; lower SDB scores do not automatically imply better clustering accuracy. K = 1 is undefined and excluded.',ha='center',fontsize=10,color='#64748b')
    output.parent.mkdir(parents=True,exist_ok=True)
    for ext in ('png','svg','pdf'):
        fig.savefig(output.with_suffix('.'+ext),dpi=220 if ext=='png' else None,facecolor='white')
        print(output.with_suffix('.'+ext),output.with_suffix('.'+ext).stat().st_size)
    plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path,default=Path('results/abstract_reproduction'))
    p.add_argument('--output',type=Path,default=Path('results/figures/figure1_independent_reproduction.png'))
    a=p.parse_args();draw(a.input_dir,a.output)

if __name__=='__main__':main()
