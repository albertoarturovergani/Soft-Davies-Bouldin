"""Compare saved two-, three-, four- and five-class sweeps on linear axes."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plot_overlap_recovery_linear import recovery_lines
from plot_supplementary_overlap import save


def draw(results, output):
    studies=[]
    for classes in [2,3,4,5]:
        directory=results/('supplementary_overlap'+(f'_{classes}classes' if classes>2 else ''))
        config=json.loads((directory/'provenance.json').read_text())['configuration']
        assert config.get('n_classes',2)==classes
        studies.append((classes,config,pd.read_csv(directory/'sweep_summary.csv')))
    reference=studies[0][1]
    for _,config,_ in studies[1:]:
        for key in ['replicates','initializations','steps','n_per_class','max_k','max_iter','tol','test_per_class']:
            assert config[key]==reference[key],f'Protocols differ: {key}'
    fig,axes=plt.subplots(2,2,figsize=(14.5,10),layout='constrained',sharey=True)
    for ax,(classes,config,data) in zip(axes.flat,studies):
        recovery_lines(ax,data,classes,config['steps'],italian=True)
        ax.set_title(f'{classes} classi · K corretto={classes} · {classes*config["n_per_class"]} osservazioni',fontweight='bold')
        ax.legend(loc='upper left',fontsize=9)
    fig.suptitle(f'Da due a cinque classi: recupero del numero di cluster\n{reference["steps"]+1} livelli · {reference["replicates"]} dataset · {reference["initializations"]} inizializzazioni · candidati K=2–{reference["max_k"]} · assi lineari',fontsize=17,fontweight='bold')
    fig.supxlabel('100% di sovrapposizione rimossa = classi disgiunte. Il punto iniziale è omesso: K=1 non è definito.\nSi mantengono 100 punti per classe; cambiano numerosità totale e geometria. Il numero di candidati sopra il vero K diminuisce.',fontsize=10)
    save(fig,output/'overlap_2_to_5_recovery')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,default=Path('results'))
    parser.add_argument('--output-dir',type=Path,default=Path('results/figures'))
    args=parser.parse_args()
    draw(args.results_dir,args.output_dir)
