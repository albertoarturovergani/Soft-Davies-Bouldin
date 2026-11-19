"""Linear hundred-feature recovery curves, projected examples and validity checks."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_overlap_recovery_linear import recovery_lines
from plot_supplementary_overlap import COLORS, LABELS, class_appearance, save
from supplementary_overlap import INDEXES, posterior_probabilities
from supplementary_overlap_hundred_features import make_data


def draw_examples(ax,config,step,legend=False):
    classes = config['n_classes']
    d = config['separations'][step]
    x,y = make_data(0,d,config['n_per_class'],n_classes=classes)
    along = x.sum(axis=1)/np.sqrt(100)
    across = (x[:,:50].sum(axis=1)-x[:,50:].sum(axis=1))/np.sqrt(100)
    colors,_ = class_appearance(classes)
    for c,color,marker in zip(range(classes),colors,['o','s','^','D','P']):
        ax.scatter(along[y==c],across[y==c],s=10,c=color,marker=marker,alpha=.5,
                   edgecolors='none',label=f'Classe {c+1}')
        ax.plot((c-(classes-1)/2)*d*np.sqrt(100),0,marker='x',color=color,ms=8,mew=2)
    ax.set(xlabel='Proiezione sulla direzione dei centri',ylabel='Contrasto ortogonale',
           title=f'Sovrapposizione 100D: {100-step*10}%',ylim=(-1.5,1.5))
    # Local x ranges show the changing geometry; ticks explicitly label scale.
    ax.grid(alpha=.15)
    if legend:
        ax.legend(fontsize=7,loc='upper left',framealpha=.9)


def draw(directory,output):
    config = json.loads((directory/'provenance.json').read_text())['configuration']
    classes = config['n_classes']
    prefix = f'overlap_100features_{classes}classes'
    summary = pd.read_csv(directory/'sweep_summary.csv')
    fig = plt.figure(figsize=(16,10),layout='constrained')
    grid = fig.add_gridspec(2,4,height_ratios=[2.1,1.25])
    ax = fig.add_subplot(grid[0,:])
    recovery_lines(ax,summary,classes,10,italian=True)
    ax.legend(loc='upper left',fontsize=10)
    top = ax.secondary_xaxis('top',functions=(lambda x:100-x,lambda x:100-x))
    top.set_xticks(np.arange(0,101,10))
    top.set_xlabel('Sovrapposizione residua delle densità congiunte (%)')
    for i,step in enumerate([0,5,8,10]):
        draw_examples(fig.add_subplot(grid[1,i]),config,step,legend=i==0)
    fig.suptitle(f'Cento feature, {classes} classi: recupero del numero di cluster\n11 livelli · 20 dataset · 100 inizializzazioni per K=2–6 · assi lineari',fontsize=17,fontweight='bold')
    fig.supxlabel('Y = frazione dei 20 dataset che selezionano il K richiesto corretto; non è accuratezza di classificazione.\nEsempi: proiezioni fisse che usano tutte le 100 feature, con scala X adattata a ciascun pannello; la proiezione non conserva la sovrapposizione 100D.\nA sovrapposizione completa K=1 non è valutabile: il punto iniziale è omesso. La validità dei cluster è verificata nel grafico dedicato.',fontsize=9.5)
    save(fig,output/(prefix+'_recovery_it'))

    selected = pd.read_csv(directory/'selected_k.csv')
    fig,axes = plt.subplots(1,3,figsize=(15,5.7),layout='constrained',sharey=True)
    for ax,metric,label in zip(axes,INDEXES,LABELS):
        data = selected[selected.metric==metric]
        counts = pd.crosstab(data.step,data.selected_k).reindex(index=range(11),columns=[0,2,3,4,5,6],fill_value=0)
        fractions = counts/config['replicates']
        im = ax.imshow(fractions,cmap='Blues',vmin=0,vmax=1,aspect='auto')
        for (i,j),value in np.ndenumerate(fractions.to_numpy()):
            if value:
                ax.text(j,i,f'{value:.0%}',ha='center',va='center',fontsize=9,color='white' if value>.6 else '#163047')
        ax.set(title=label,xlabel='K richiesto selezionato',xticks=range(6),xticklabels=['Nessuno',2,3,4,5,6],
               yticks=range(11),yticklabels=[f'{100-10*s}%' for s in range(11)])
    axes[0].set_ylabel('Sovrapposizione residua congiunta')
    fig.colorbar(im,ax=axes,shrink=.8,label='Frazione dei dataset')
    fig.suptitle(f'Cento feature · {classes} classi reali · distribuzione del K selezionato\nNessuno = tutti i candidati hanno una media non finita; il livello 100% è solo descrittivo',fontsize=15,fontweight='bold')
    save(fig,output/(prefix+'_selected_k'))

    fig,axes = plt.subplots(1,2,figsize=(14,5.8),layout='constrained')
    for metric,label,color in zip(INDEXES,LABELS,COLORS):
        a = summary[(summary.metric==metric)&(summary.step>0)]
        axes[0].plot(a.step*10,a.distinct_clusters_correct_rate,'o-',color=color,label=label)
        axes[1].plot(a.step*10,a.distinct_clusters_no_candidate_rate,'o-',color=color,label=label)
    for ax in axes:
        ax.set(xlim=(-2,102),ylim=(-.03,1.08),xticks=np.arange(0,101,10),xlabel='Sovrapposizione rimossa (%)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
    axes[0].set(title=f'Scelta di K={classes} dopo il controllo di validità',ylabel='Frazione dei dataset')
    axes[1].set(title='Nessun candidato supera il controllo di validità',ylabel='Frazione dei dataset')
    fig.suptitle(f'Cento feature · {classes} classi · sensibilità ai centri quasi coincidenti',fontsize=16,fontweight='bold')
    fig.supxlabel('Diagnostica aggiuntiva: un candidato FCM è escluso se almeno una delle 100 inizializzazioni ha distanza minima tra centri / RMS dei dati < 10⁻⁶.\nPer DB/FCM si richiedono anche K gruppi non vuoti in ogni inizializzazione. Il criterio pubblicato e i risultati primari restano invariati.',fontsize=9.5)
    save(fig,output/(prefix+'_validity'))

    perf = pd.read_csv(directory/f'k{classes}_performance.csv')
    deg = pd.read_csv(directory/'degeneracy_by_step.csv')
    fig,axes = plt.subplots(1,2,figsize=(14,5.8),layout='constrained')
    for method,color,style in [('KMeans',COLORS[0],'-'),('FCM',COLORS[2],'--')]:
        a = perf[perf.method==method]
        axes[0].plot(a.step*10,a.test_accuracy_mean,marker='o',color=color,ls=style,label=method)
    axes[0].plot(a.step*10,a.bayes_accuracy,'k:',label='Riferimento Bayes')
    axes[0].set(title=f'Accuratezza su dati indipendenti, K={classes} fissato',ylabel='Accuratezza media')
    axes[1].plot(deg.step*10,deg.near_coincident_fcm_fits/deg.configurations,'o-',label='Centri FCM quasi coincidenti')
    axes[1].plot(deg.step*10,deg.hard_cluster_count_below_requested/deg.configurations,'s--',label='FCM: gruppi netti < K richiesto')
    axes[1].plot(deg.step*10,deg.SDB_fcm_undefined/deg.configurations,'D-',label='SDB non definito')
    axes[1].set(title='Diagnostica su tutti i K e tutte le inizializzazioni',ylabel='Frazione delle configurazioni')
    for ax in axes:
        ax.set(xlim=(-2,102),ylim=(-.03,1.05),xticks=np.arange(0,101,10),xlabel='Sovrapposizione rimossa (%)')
        ax.legend(fontsize=9)
        ax.grid(alpha=.2)
    fig.suptitle(f'Cento feature · {classes} classi: assegnazione dei punti e degenerazione delle soluzioni',fontsize=15,fontweight='bold')
    fig.supxlabel('Accuratezza: per ogni dataset e K, partizione con il migliore obiettivo nativo tra 100 inizializzazioni.\nLa scelta automatica di K usa invece la media dell’indice su tutte le inizializzazioni.',fontsize=10)
    save(fig,output/(prefix+'_performance'))

    examples = []
    for step,d in enumerate(config['separations']):
        x,y = make_data(0,d,config['n_per_class'],n_classes=classes)
        record = dict(step=step,overlap_pct=100-10*step,**{'class':y},**{f'x{i+1}':x[:,i] for i in range(100)})
        posterior = posterior_probabilities(x,d,classes,100)
        record.update({f'posterior_class_{i}':posterior[:,i] for i in range(classes)})
        examples.append(pd.DataFrame(record))
    pd.concat(examples).to_csv(directory/'example_datasets.csv',index=False,float_format='%.12g')


def compare_all(results,output):
    fig,axes = plt.subplots(2,2,figsize=(14.5,10),layout='constrained',sharey=True)
    for ax,classes in zip(axes.flat,range(2,6)):
        summary = pd.read_csv(results/f'supplementary_overlap_100features_{classes}classes/sweep_summary.csv')
        recovery_lines(ax,summary,classes,10,italian=True)
        ax.set_title(f'{classes} classi · 100 feature · {classes*100} osservazioni',fontweight='bold')
        ax.legend(fontsize=9,loc='upper left')
    fig.suptitle('Cento feature: recupero del numero di cluster da due a cinque classi\n20 dataset · 100 inizializzazioni per K=2–6 · assi lineari',fontsize=17,fontweight='bold')
    fig.supxlabel('Frequenza di selezione del K richiesto: consultare anche la diagnostica dei centri quasi coincidenti.\nIl livello di sovrapposizione completa è omesso perché K=1 non è definito.',fontsize=10)
    save(fig,output/'overlap_100features_2_to_5_recovery')
    fig,axes = plt.subplots(4,5,figsize=(28,17),layout='constrained',sharex=True,sharey=True)
    for row,classes in enumerate(range(2,6)):
        for col,features in enumerate([1,2,3,10,100]):
            suffix = f'_{features}features_{classes}classes' if features>1 else (f'_{classes}classes' if classes>2 else '')
            path = results/('supplementary_overlap'+suffix)/'sweep_summary.csv'
            ax = axes[row,col]
            ax.set_title(f'{classes} classi · {features} feature',fontweight='bold')
            if path.exists():
                recovery_lines(ax,pd.read_csv(path),classes,10,italian=True)
                if row==0:
                    ax.legend(fontsize=8,loc='upper left')
            else:
                ax.text(.5,.5,'Test interrotto\nRisultato non disponibile',transform=ax.transAxes,ha='center',va='center',fontsize=13)
                ax.set_facecolor('#f4f4f4')
    fig.suptitle('Una, due, tre, dieci e cento feature: confronto a pari sovrapposizione congiunta\n20 dataset · 100 inizializzazioni · geometria con centri allineati · assi lineari',fontsize=17,fontweight='bold')
    fig.supxlabel('In 100D compaiono centri FCM quasi coincidenti: il K richiesto può non corrispondere a cluster distinti.\nConfronto descrittivo: numero di feature, geometria delle distanze e comportamento del solver cambiano insieme.',fontsize=10)
    save(fig,output/'overlap_1_vs_2_vs_3_vs_10_vs_100_features')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path)
    parser.add_argument('--output-dir',type=Path,default=Path('results/figures'))
    parser.add_argument('--results-dir',type=Path,default=Path('results'))
    parser.add_argument('--compare-all',action='store_true')
    args = parser.parse_args()
    if args.compare_all:
        compare_all(args.results_dir,args.output_dir)
    else:
        draw(args.input_dir,args.output_dir)
