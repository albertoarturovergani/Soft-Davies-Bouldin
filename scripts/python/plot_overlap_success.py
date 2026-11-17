"""Focused Italian figure: separation continuum versus correct class-count recovery."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from supplementary_overlap import density, make_data, separation_for_step


def draw(directory, output):
    data = pd.read_csv(directory/'sweep_summary.csv')
    config = json.loads((directory/'provenance.json').read_text())['configuration']
    max_step = int(data.step.max())
    example_steps = [0,round(.5*max_step),round(.8*max_step),max_step]
    fig = plt.figure(figsize=(14.4,10.8),layout='constrained')
    gs = fig.add_gridspec(3,4,height_ratios=[3.7,.36,1.45])
    ax = fig.add_subplot(gs[0,:])
    zero_ax = fig.add_subplot(gs[1,:],sharex=ax)
    styles = [('DB_kmeans','k-means + Davies–Bouldin','#2563a6','o','-',12),
              ('DB_fcm_hard','Fuzzy c-means + Davies–Bouldin','#b87417','s','--',8),
              ('SDB_fcm','Fuzzy c-means + Soft Davies–Bouldin','#ae2855','D','-',5)]
    for metric,label,color,marker,linestyle,size in styles:
        rows=data[(data.metric==metric)&(data.step>0)].sort_values('step')
        positive = np.where(rows.selected_k2_rate>0,rows.selected_k2_rate,np.nan)
        ax.plot(2*rows.step/max_step,positive,label=label,color=color,
                marker=marker,linestyle=linestyle,lw=2,ms=size,markerfacecolor='white',markeredgewidth=1.6)
        zeros = rows.loc[rows.selected_k2_rate==0,'step']
        zero_ax.plot(2*zeros/max_step,np.zeros(len(zeros)),linestyle='none',color=color,
                     marker=marker,ms=size,markerfacecolor='white',markeredgewidth=1.6)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(.9*2/max_step,2.08)
    ax.set_ylim(.1,1.17)
    ax.set_xticks([.2,.4,.6,1,1.6,2],labels=['0,2','0,4','0,6','1','1,6','2'])
    ax.minorticks_off()
    ax.set_yticks([.1,.15,.25,.5,.75,1],labels=['0,10','0,15','0,25','0,50','0,75','1,00'])
    ax.tick_params(axis='x',labelbottom=False)
    ax.set_ylabel('Frequenza di recupero del numero di classi · scala log\n1 = tutti i 20 dataset',fontsize=12)
    ax.grid(alpha=.2)
    top=ax.secondary_xaxis('top',functions=(lambda x:100*(1-x/2),lambda overlap:2*(1-overlap/100)))
    top.set_xticks([90,80,60,40,20,10,0],labels=['90','80','60','40','20','10','0'])
    top.minorticks_off()
    top.set_xlabel('Sovrapposizione residua delle distribuzioni (%)',fontsize=11,labelpad=8)
    ax.text(.035,.40,'Entrambi gli assi principali sono logaritmici.\nI successi Y=0 sono nella fascia separata qui sotto.\nX=0 è mostrato nel primo esempio di dataset;\nlì servirebbe K=1, non valutabile con questi indici.',transform=ax.transAxes,fontsize=10,color='#475569',bbox=dict(boxstyle='round,pad=.5',fc='#f1f5f9',ec='#cbd5e1'))
    ax.annotate('10% di sovrapposizione\nSDB: 19/20 = 0,95\nDB/k-means: 16/20 = 0,80\nDB/FCM: 15/20 = 0,75',xy=(1.8,.95),xytext=(.64,.39),fontsize=10,arrowprops=dict(arrowstyle='->',color='#ae2855'),bbox=dict(boxstyle='round,pad=.5',fc='#fff7fa',ec='#e5b5c6'))
    ax.legend(loc='upper left',fontsize=10,framealpha=1)
    zero_ax.set(ylim=(-.6,.6),yticks=[0],yticklabels=['0'])
    zero_ax.set_facecolor('#f1f5f9')
    zero_ax.grid(axis='x',alpha=.15)
    zero_ax.set_ylabel('Y=0\nfuori scala',fontsize=10)
    zero_ax.set_xlabel('Continuum di separazione · scala log X: x = 2 × (1 − quota sovrapposta)\nLa fuzziness diminuisce verso destra; x=2 corrisponde a classi esclusive',fontsize=12,labelpad=8)
    fig.suptitle('Recupero delle classi · grafico log–log\n11 livelli · 20 dataset indipendenti · 100 inizializzazioni · candidati K=2–6',fontsize=16,fontweight='bold')
    grid = np.linspace(-2.1,2.1,1001)
    for position,step in enumerate(example_steps):
        bx = fig.add_subplot(gs[2,position])
        d=separation_for_step(step,max_step)
        f0,f1=density(grid,-d/2),density(grid,d/2)
        bx.fill_between(grid,np.minimum(f0,f1),color='#aa91b8',alpha=.5,label='Sovrapposizione')
        bx.plot(grid,f0,color='#2563a6',lw=1.6,label='Classe 0')
        bx.plot(grid,f1,color='#ae2855',lw=1.6,ls='--',label='Classe 1')
        samples,truth=make_data(0,d,config['n_per_class'])
        bx.plot(samples[truth==0,0],np.full((truth==0).sum(),-.06),'|',color='#2563a6',alpha=.35,ms=4)
        bx.plot(samples[truth==1,0],np.full((truth==1).sum(),-.13),'|',color='#ae2855',alpha=.35,ms=4)
        bx.set(title=f'Passo {step} · sovrapposizione {100*(1-step/max_step):.0f}%\nx = {2*step/max_step:.1f}',xlim=(-2.1,2.1),ylim=(-.18,1.05),xticks=[-2,0,2],yticks=[0,1])
        bx.grid(alpha=.15)
        bx.set_xlabel('Variabile osservata',fontsize=9)
        if position==0:
            bx.set_ylabel('Densità',fontsize=10)
            bx.legend(fontsize=8,loc='upper right')
    fig.supxlabel('Esempi dal dataset n. 1: linee = densità di classe; area viola = sovrapposizione; trattini = osservazioni.\nI quattro riquadri inferiori usano assi lineari identici per confrontare la forma dei dataset.',fontsize=10)
    output.parent.mkdir(parents=True,exist_ok=True)
    for ext in ['png','svg','pdf']:
        fig.savefig(output.with_suffix('.'+ext),dpi=180,facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=Path('results/supplementary_overlap'))
    parser.add_argument('--output',type=Path,default=Path('results/figures/overlap_success_it.png'))
    args=parser.parse_args()
    draw(args.input_dir,args.output)
