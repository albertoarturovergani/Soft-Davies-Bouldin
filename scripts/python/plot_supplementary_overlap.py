"""Publication figures for the independent 1D overlap supplementary analysis."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from supplementary_overlap import INDEXES, density, make_data, separation_for_step, true_posterior

COLORS = ['#2563a6', '#b87417', '#ae2855']
LABELS = ['DB / k-means', 'DB / hardened FCM', 'SDB / FCM']


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ['png', 'pdf', 'svg']:
        fig.savefig(path.with_suffix('.'+suffix), dpi=180, facecolor='white')
    plt.close(fig)


def draw(directory, output):
    config = json.loads((directory/'provenance.json').read_text())['configuration']
    steps = config['steps']
    examples = []
    for step in range(steps+1):
        d = separation_for_step(step, steps)
        observations, labels = make_data(0, d, config['n_per_class'])
        examples.append(pd.DataFrame({'step':step, 'overlap_pct':100*(1-step/steps),
                                      'x':observations[:,0], 'class':labels,
                                      'posterior_class_1':true_posterior(observations[:,0],d)}))
    pd.concat(examples).to_csv(directory/'example_datasets.csv',index=False,float_format='%.12g')
    x = np.linspace(-2.1, 2.1, 2001)
    rows = int(np.ceil((steps+2)/4))
    fig, axes = plt.subplots(rows, 4, figsize=(15, 2.86*rows), layout='constrained', sharex=True, sharey=True, squeeze=False)
    for step, ax in enumerate(axes.flat):
        if step > steps:
            ax.axis('off')
            if step == steps+1:
                ax.text(.03, .88, f'One observed variable\nEqual class priors\nTriangular class densities\n\nShaded area = overlap\nStep 0: indistinguishable classes\nStep {steps}: exclusive almost surely', transform=ax.transAxes, va='top', fontsize=11, linespacing=1.5)
            continue
        d = separation_for_step(step, steps)
        a, b = density(x, -d/2), density(x, d/2)
        ax.fill_between(x, np.minimum(a,b), color='#aa91b8', alpha=.55, label='Common area')
        ax.plot(x, a, color=COLORS[0], lw=1.8, label='Class 0')
        ax.plot(x, b, color=COLORS[2], lw=1.8, linestyle='--', label='Class 1')
        ax.set(title=f'Step {step} · overlap {100*(1-step/steps):.0f}%', xlim=(-2.1,2.1), ylim=(0,1.08))
        ax.grid(alpha=.15)
        if step % 4 == 0: ax.set_ylabel('Class density')
        if step >= (rows-1)*4: ax.set_xlabel('Observed variable x')
        if step == 0: ax.legend(fontsize=8, loc='lower left')
    fig.suptitle(f'Supplementary analysis · {steps+1} exact overlap levels', fontsize=18, fontweight='bold')
    save(fig, output/'overlap_distributions')

    summary = pd.read_csv(directory/'sweep_summary.csv')
    perf = pd.read_csv(directory/'k2_performance.csv')
    fig, axes = plt.subplots(2,3,figsize=(15,8.7),layout='constrained')
    for idx, (metric, label, color) in enumerate(zip(INDEXES, LABELS, COLORS)):
        a = summary[summary.metric == metric].sort_values('step')
        ax = axes[0,idx]
        ax.plot(a.step, a.score_k2_mean, 'o-', color=color, ms=4)
        ax.fill_between(a.step, a.score_k2_p10, a.score_k2_p90, color=color, alpha=.16)
        ax.set(title=f'{label} · fixed K=2', ylabel='Index value (own scale)')
        axes[1,1].plot(a.step, a.selected_k2_rate, 'o-', color=color, label=label, ms=4)
    for method, color, linestyle in [('KMeans',COLORS[0],'-'),('FCM',COLORS[2],'--')]:
        a = perf[perf.method == method].sort_values('step')
        for ax, column in [(axes[1,0],'test_accuracy'),(axes[1,2],'posterior_MSE')]:
            ax.plot(a.step,a[column+'_mean'],marker='o',linestyle=linestyle,color=color,label=method,ms=4)
            ax.fill_between(a.step,a[column+'_p10'],a[column+'_p90'],color=color,alpha=.13)
    a = perf[perf.method == 'FCM'].sort_values('step')
    axes[1,0].plot(a.step,a.bayes_accuracy,'k:',lw=1.8,label='Bayes reference')
    axes[1,0].set(title='Class recovery · held-out data, K=2',ylabel='Accuracy',ylim=(.45,1.03))
    axes[1,1].set(title='K selection · minimize mean index',ylabel='Fraction selecting K=2',ylim=(-.03,1.05))
    axes[1,2].set(title='Membership vs known posterior · K=2',ylabel='Mean squared error (lower is better)',ylim=(-.01,.27))
    for ax in axes.flat:
        ax.set_xlabel('Separation step (0–10)')
        ax.set_xticks(range(steps+1))
        ax.grid(alpha=.2)
    for ax in axes[1]: ax.legend(fontsize=8,loc='best')
    fig.suptitle(f'Supplementary overlap sweep · {config["replicates"]} datasets × {config["initializations"]} initializations per K\n100%, 90%, …, 0% overlap · bands: 10th–90th percentiles across datasets',fontsize=16,fontweight='bold')
    save(fig, output/'overlap_results')

    selected = pd.read_csv(directory/'selected_k.csv')
    fig, axes = plt.subplots(1,3,figsize=(14,5.4),layout='constrained',sharey=True)
    for ax, metric, label in zip(axes,INDEXES,LABELS):
        a = selected[selected.metric == metric]
        counts = pd.crosstab(a.step,a.selected_k).reindex(index=range(steps+1),columns=range(2,config['max_k']+1),fill_value=0)
        frequencies = counts / config['replicates']
        im = ax.imshow(frequencies,cmap='Blues',vmin=0,vmax=1,aspect='auto')
        for (y,z),v in np.ndenumerate(frequencies.to_numpy()):
            if v: ax.text(z,y,f'{int(round(v*100))}%',ha='center',va='center',fontsize=9,color='white' if v>.6 else '#163047')
        ax.set(title=label,xlabel='Selected K',xticks=range(config['max_k']-1),xticklabels=range(2,config['max_k']+1),yticks=range(steps+1),yticklabels=[f'{s} ({100*(1-s/steps):.0f}%)' for s in range(steps+1)])
    axes[0].set_ylabel('Step (remaining overlap)')
    fig.colorbar(im,ax=axes,shrink=.8,label='Fraction of independent datasets')
    fig.suptitle('Cluster-count selection across the overlap continuum\nK=1 is undefined for all three indices; step 0 has no identifiable two-class structure',fontsize=15,fontweight='bold')
    save(fig,output/'overlap_selected_k')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=Path('results/supplementary_overlap'))
    parser.add_argument('--output-dir',type=Path,default=Path('results/figures'))
    args = parser.parse_args()
    draw(args.input_dir,args.output_dir)
