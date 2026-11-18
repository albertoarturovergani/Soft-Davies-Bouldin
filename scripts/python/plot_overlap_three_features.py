"""Linear recovery curves, 3D examples and matched-overlap dimension comparison."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_overlap_recovery_linear import recovery_lines
from plot_supplementary_overlap import draw_three_feature_cloud,save


def draw(directory,output):
    config=json.loads((directory/'provenance.json').read_text())['configuration']
    assert config['n_features']==3
    classes=config['n_classes']
    summary=pd.read_csv(directory/'sweep_summary.csv')
    fig=plt.figure(figsize=(16,10.4),layout='constrained')
    grid=fig.add_gridspec(2,4,height_ratios=[2.2,1.4])
    ax=fig.add_subplot(grid[0,:])
    recovery_lines(ax,summary,classes,config['steps'],italian=True)
    ax.legend(loc='upper left',fontsize=10)
    top=ax.secondary_xaxis('top',functions=(lambda x:100-x,lambda x:100-x))
    top.set_xticks(np.arange(0,101,10))
    top.set_xlabel('Sovrapposizione congiunta residua tra classi adiacenti (%)')
    for position,step in enumerate([0,round(.5*config['steps']),round(.8*config['steps']),config['steps']]):
        draw_three_feature_cloud(fig.add_subplot(grid[1,position],projection='3d'),config,step,legend=position==0)
    fig.suptitle(f'Tre feature, {classes} classi: quanto spesso viene recuperato K={classes}?\n{config["steps"]+1} livelli · {config["replicates"]} dataset · {config["initializations"]} inizializzazioni · candidati K=2–{config["max_k"]}',fontsize=17,fontweight='bold')
    fig.supxlabel('Assi lineari. Y = frazione dei dataset con K corretto. Il punto iniziale è omesso: K=1 non è definito.\nEsempi 3D: punti = osservazioni; cubi = limiti dei supporti; croci = centri. Sovrapposizione delle densità congiunte.\nTre feature informative, rumore indipendente entro classe, centri lungo la diagonale; stessa scala nei quattro esempi.',fontsize=10)
    save(fig,output/f'overlap_3features_{classes}classes_recovery_it')


def compare_all(results,output):
    studies={}
    for classes in range(2,6):
        for features in [1,2,3]:
            suffix=f'_{features}features_{classes}classes' if features>1 else (f'_{classes}classes' if classes>2 else '')
            directory=results/('supplementary_overlap'+suffix)
            config=json.loads((directory/'provenance.json').read_text())['configuration']
            assert config.get('n_features',1)==features
            studies[classes,features]=(config,pd.read_csv(directory/'sweep_summary.csv'))
    reference=studies[2,1][0]
    for config,_ in studies.values():
        for key in ['replicates','initializations','steps','n_per_class','test_per_class','max_k','max_iter','tol']:
            assert config[key]==reference[key],f'Protocol mismatch: {key}'
    fig,axes=plt.subplots(2,2,figsize=(14.5,10),layout='constrained',sharey=True)
    for ax,classes in zip(axes.flat,range(2,6)):
        config,summary=studies[classes,3]
        recovery_lines(ax,summary,classes,config['steps'],italian=True)
        ax.set_title(f'{classes} classi · 3 feature · {classes*config["n_per_class"]} osservazioni',fontweight='bold')
        ax.legend(loc='upper left',fontsize=9)
    fig.suptitle('Tre feature: confronto da due a cinque classi\n11 livelli · 20 dataset · 100 inizializzazioni per K=2–6 · assi lineari',fontsize=17,fontweight='bold')
    fig.supxlabel('100% di sovrapposizione rimossa = classi disgiunte. Il punto iniziale è omesso: K=1 non è definito.\nSovrapposizione congiunta per coppia adiacente; 100 punti per classe, centri lungo la diagonale.',fontsize=10)
    save(fig,output/'overlap_3features_2_to_5_recovery')

    fig,axes=plt.subplots(4,3,figsize=(19,17),layout='constrained',sharex=True,sharey=True)
    for row,classes in enumerate(range(2,6)):
        for column,features in enumerate([1,2,3]):
            config,summary=studies[classes,features]
            ax=axes[row,column]
            recovery_lines(ax,summary,classes,config['steps'],italian=True)
            ax.set_title(f'{classes} classi · {features} feature · {classes*config["n_per_class"]} osservazioni',fontweight='bold')
            if row==0:
                ax.legend(loc='upper left',fontsize=8)
    fig.suptitle('Una, due o tre feature: stessa sovrapposizione congiunta\n20 dataset · 100 inizializzazioni per K=2–6 · assi lineari',fontsize=17,fontweight='bold')
    fig.supxlabel('Confronto a pari sovrapposizione delle densità: cambiano la geometria delle distanze e le membership fuzzy.\nFeature informative con rumore indipendente entro classe; centri lungo la diagonale.\nIl punto a sovrapposizione completa è omesso: K=1 non è definito.',fontsize=10)
    save(fig,output/'overlap_1_vs_2_vs_3_features')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=Path('results/supplementary_overlap_3features_2classes'))
    parser.add_argument('--results-dir',type=Path,default=Path('results'))
    parser.add_argument('--output-dir',type=Path,default=Path('results/figures'))
    parser.add_argument('--compare-all',action='store_true')
    args=parser.parse_args()
    if args.compare_all:
        compare_all(args.results_dir,args.output_dir)
    else:
        draw(args.input_dir,args.output_dir)
