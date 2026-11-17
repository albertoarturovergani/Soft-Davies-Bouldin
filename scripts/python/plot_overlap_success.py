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
    panel_rows = int(np.ceil((max_step+2)/6))
    fig = plt.figure(figsize=(14.4, 7+2.1*panel_rows),layout='constrained')
    gs = fig.add_gridspec(1+panel_rows,6,height_ratios=[3.7]+[1.2]*panel_rows)
    ax = fig.add_subplot(gs[0,:])
    styles = [('DB_kmeans','k-means + Davies–Bouldin','#2563a6','o','-',12),
              ('DB_fcm_hard','Fuzzy c-means + Davies–Bouldin','#b87417','s','--',8),
              ('SDB_fcm','Fuzzy c-means + Soft Davies–Bouldin','#ae2855','D','-',5)]
    for metric,label,color,marker,linestyle,size in styles:
        rows=data[(data.metric==metric)&(data.step>0)].sort_values('step')
        ax.plot(2*rows.step/max_step,rows.selected_k2_rate,label=label,color=color,
                marker=marker,linestyle=linestyle,lw=2,ms=size,markerfacecolor='white',markeredgewidth=1.6)
    ax.set_xscale('log')
    ax.set_xlim(.9*2/max_step,2.08)
    ax.set_ylim(-.03,1.06)
    ax.set_xticks([.2,.4,.6,1,1.6,2],labels=['0,2','0,4','0,6','1','1,6','2'])
    ax.minorticks_off()
    ax.set_yticks(np.linspace(0,1,6))
    ax.set_xlabel('Continuum di separazione in scala logaritmica: x = 2 × (1 − sovrapposizione)\nLa fuzziness diminuisce verso destra; x=2 corrisponde a classi esclusive',fontsize=12,labelpad=10)
    ax.set_ylabel('Frequenza di recupero del numero di classi\n0 = nessun dataset · 1 = tutti i 20 dataset',fontsize=12)
    ax.grid(alpha=.2)
    top=ax.secondary_xaxis('top',functions=(lambda x:100*(1-x/2),lambda overlap:2*(1-overlap/100)))
    top.set_xticks([90,80,60,40,20,10,0],labels=['90','80','60','40','20','10','0'])
    top.minorticks_off()
    top.set_xlabel('Sovrapposizione residua delle distribuzioni (%)',fontsize=11,labelpad=8)
    ax.text(.035,.47,'Lo zero non esiste sulla scala logaritmica.\nLa sovrapposizione completa è illustrata nel riquadro 0.\nLì K=1 sarebbe corretto, ma gli indici non lo valutano.',transform=ax.transAxes,fontsize=10,color='#475569',bbox=dict(boxstyle='round,pad=.5',fc='#f1f5f9',ec='#cbd5e1'))
    ax.annotate('10% di sovrapposizione\nSDB: 19/20 = 0,95\nDB/k-means: 16/20 = 0,80\nDB/FCM: 15/20 = 0,75',xy=(1.8,.95),xytext=(.64,.58),fontsize=10,arrowprops=dict(arrowstyle='->',color='#ae2855'),bbox=dict(boxstyle='round,pad=.5',fc='#fff7fa',ec='#e5b5c6'))
    ax.legend(loc='upper left',fontsize=10,framealpha=1)
    fig.suptitle('Quando vengono riconosciute le due classi?\n11 livelli · 20 dataset indipendenti · 100 inizializzazioni · candidati K=2–6',fontsize=16,fontweight='bold')
    grid = np.linspace(-2.1,2.1,1001)
    for step in range(panel_rows*6):
        bx = fig.add_subplot(gs[1+step//6,step%6])
        if step>max_step:
            bx.axis('off')
            if step==max_step+1:
                bx.text(.02,.95,'Esempio: dataset n. 1\n\nBlu: classe generativa 0\nRosa: classe generativa 1\nArea viola: sovrapposizione\nTrattini: campioni osservati\n\nI riquadri inferiori hanno\nassi x lineari e identici.',va='top',fontsize=10,transform=bx.transAxes,linespacing=1.35)
            continue
        d=separation_for_step(step,max_step)
        f0,f1=density(grid,-d/2),density(grid,d/2)
        bx.fill_between(grid,np.minimum(f0,f1),color='#aa91b8',alpha=.5)
        bx.plot(grid,f0,color='#2563a6',lw=1.6)
        bx.plot(grid,f1,color='#ae2855',lw=1.6,ls='--')
        samples,truth=make_data(0,d,config['n_per_class'])
        bx.plot(samples[truth==0,0],np.full((truth==0).sum(),-.06),'|',color='#2563a6',alpha=.35,ms=4)
        bx.plot(samples[truth==1,0],np.full((truth==1).sum(),-.13),'|',color='#ae2855',alpha=.35,ms=4)
        bx.set(title=f'Passo {step} · overlap {100*(1-step/max_step):.0f}%\nx = {2*step/max_step:.1f}',xlim=(-2.1,2.1),ylim=(-.18,1.05),xticks=[-2,0,2],yticks=[0,1])
        bx.grid(alpha=.15)
        bx.set_xlabel('Variabile osservata',fontsize=9)
        if step%6==0: bx.set_ylabel('Densità',fontsize=10)
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
