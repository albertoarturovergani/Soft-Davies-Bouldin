"""Linear-axis recovery plots for the two- and three-class overlap experiments."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_supplementary_overlap import COLORS, LABELS, save
from supplementary_overlap import INDEXES, density, generating_centers, make_data, separation_for_step


def recovery_lines(ax, data, n_classes, steps, *, italian=False):
    names = ['DB / k-means', 'DB / FCM con assegnazioni nette', 'SDB / FCM'] if italian else LABELS
    for metric,label,color,marker,style in zip(INDEXES,names,COLORS,['o','s','D'],['-','--','-']):
        group=data[(data.metric==metric)&(data.step>0)].sort_values('step')
        ax.plot(100*group.step/steps,group[f'selected_k{n_classes}_rate'],color=color,
                marker=marker,linestyle=style,markerfacecolor='white',ms=5,lw=1.8,label=label)
    ax.set(xlim=(-2,102),ylim=(-.03,1.08),xticks=np.arange(0,101,10),yticks=np.arange(0,1.01,.2))
    ax.grid(alpha=.2)
    ax.set_xlabel('Sovrapposizione rimossa (%)' if italian else 'Overlap removed (%)')
    ax.set_ylabel(f'Frequenza di scelta corretta di K={n_classes}' if italian else f'Fraction selecting K={n_classes}')


def draw(directory, output, baseline):
    config=json.loads((directory/'provenance.json').read_text())['configuration']
    n_classes=config.get('n_classes',2)
    steps=config['steps']
    data=pd.read_csv(directory/'sweep_summary.csv')
    fig=plt.figure(figsize=(14.4,9.1),layout='constrained')
    gs=fig.add_gridspec(2,4,height_ratios=[2.8,1.2])
    ax=fig.add_subplot(gs[0,:])
    recovery_lines(ax,data,n_classes,steps,italian=True)
    ax.legend(loc='upper left',fontsize=10)
    top=ax.secondary_xaxis('top',functions=(lambda x:100-x,lambda x:100-x))
    top.set_xticks(np.arange(0,101,10))
    top.set_xlabel('Sovrapposizione residua tra classi adiacenti (%)')
    ax.text(.025,.38,'Assi lineari: 0 significa nessun dataset, 1 tutti i dataset.\nA sovrapposizione completa resta una sola distribuzione:\nK=1 non è valutabile, quindi il punto iniziale è omesso.',transform=ax.transAxes,
            fontsize=10,color='#475569',bbox=dict(boxstyle='round,pad=.5',facecolor='#f1f5f9',edgecolor='#cbd5e1'))
    fig.suptitle(f'Tre classi: quanto spesso viene recuperato K={n_classes}?\n{steps+1} livelli · {config["replicates"]} dataset · {config["initializations"]} inizializzazioni · candidati K=2–{config["max_k"]}',fontsize=17,fontweight='bold')
    grid=np.linspace(-n_classes-.1,n_classes+.1,1401)
    for position,step in enumerate([0,round(.5*steps),round(.8*steps),steps]):
        bx=fig.add_subplot(gs[1,position])
        separation=separation_for_step(step,steps)
        curves=[density(grid,center) for center in generating_centers(separation,n_classes)]
        for j in range(n_classes-1):
            bx.fill_between(grid,np.minimum(curves[j],curves[j+1]),alpha=.25,color='#aa91b8')
        observations,truth=make_data(0,separation,config['n_per_class'],n_classes=n_classes)
        for j,(curve,color,style) in enumerate(zip(curves,COLORS,['-','--',':'])):
            bx.plot(grid,curve,color=color,ls=style,lw=1.8,label=f'Classe {j+1}')
            bx.plot(observations[truth==j,0],np.full((truth==j).sum(),-.06-.07*j),'|',color=color,alpha=.35,ms=4)
        bx.set(title=f'Passo {step}: sovrapposizione {100*(1-step/steps):.0f}%',
               xlim=(-n_classes-.1,n_classes+.1),ylim=(-.27,1.05),xticks=[-n_classes,0,n_classes],yticks=[0,1],xlabel='Variabile osservata')
        bx.grid(alpha=.15)
        if position==0:
            bx.set_ylabel('Densità')
            bx.legend(fontsize=8,loc='upper right')
    fig.supxlabel('Esempi dal primo dataset: linee = densità di classe; ombreggiatura = sovrapposizioni adiacenti; trattini = osservazioni.\nLa sovrapposizione percentuale si riferisce a ciascuna coppia adiacente, non all’intersezione di tutte e tre le classi.',fontsize=10)
    save(fig,output/f'overlap_{n_classes}classes_recovery_it')

    original=pd.read_csv(baseline/'sweep_summary.csv')
    original_config=json.loads((baseline/'provenance.json').read_text())['configuration']
    fig,axes=plt.subplots(1,2,figsize=(13,5.8),layout='constrained',sharey=True)
    for bx,table,cfg in [(axes[0],original,original_config),(axes[1],data,config)]:
        classes=cfg.get('n_classes',2)
        recovery_lines(bx,table,classes,cfg['steps'])
        bx.set_title(f'{classes} generating classes · {classes*cfg["n_per_class"]} training samples',fontweight='bold')
        bx.legend(loc='upper left',fontsize=9)
    fig.suptitle('Does the two-class K-selection advantage extend to three classes?\nEqual samples per class, candidate K=2–6, 20 datasets and 100 starts per K; linear axes',fontsize=14,fontweight='bold')
    fig.supxlabel('The 100%-overlap endpoint is omitted: K=1 is undefined. Overlap means adjacent-pair density overlap.\nChanging the class count also changes total sample size and mixture geometry; this is a descriptive comparison.',fontsize=10)
    save(fig,output/'overlap_2_vs_3_recovery')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=Path('results/supplementary_overlap_3classes'))
    parser.add_argument('--baseline-dir',type=Path,default=Path('results/supplementary_overlap'))
    parser.add_argument('--output-dir',type=Path,default=Path('results/figures'))
    args=parser.parse_args()
    draw(args.input_dir,args.output_dir,args.baseline_dir)
