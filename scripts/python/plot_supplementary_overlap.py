"""Publication figures for the independent 1D/2D/3D overlap analysis."""
import argparse
import json
from itertools import product, combinations
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from supplementary_overlap import INDEXES, density, generating_centers, make_data, posterior_probabilities, separation_for_step

COLORS = ['#2563a6', '#b87417', '#ae2855']
LABELS = ['DB / k-means', 'DB / hardened FCM', 'SDB / FCM']


def class_appearance(n_classes):
    colors = [COLORS[0],COLORS[2]] if n_classes==2 else COLORS+['#198568','#7854aa']
    styles = ['-','--',':','-.',(0,(5,1,1,1,1,1))]
    return colors[:n_classes], styles[:n_classes]


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ['png', 'pdf', 'svg']:
        fig.savefig(path.with_suffix('.'+suffix), dpi=180, facecolor='white')
    plt.close(fig)


def draw_two_feature_cloud(ax, config, step, *, legend=False):
    classes=config['n_classes']
    d=separation_for_step(step,config['steps'],2)
    x,labels=make_data(0,d,config['n_per_class'],n_classes=classes,n_features=2)
    colors,_=class_appearance(classes)
    for j,(center,color,marker) in enumerate(zip(generating_centers(d,classes),colors,['o','s','^','D','P'])):
        subset=x[labels==j]
        ax.scatter(subset[:,0],subset[:,1],s=10,marker=marker,c=color,alpha=.5,edgecolors='none',label=f'Classe {j+1}')
        ax.add_patch(Rectangle((center-1,center-1),2,2,fill=False,edgecolor=color,linewidth=.8,alpha=.65))
        ax.plot(center,center,marker='x',color=color,ms=6)
    ax.set(xlim=(-classes-.1,classes+.1),ylim=(-classes-.1,classes+.1),
           xlabel='Feature 1',ylabel='Feature 2',title=f'Passo {step}: sovrapposizione {100*(1-step/config["steps"]):.0f}%')
    ax.set_aspect('equal',adjustable='box')
    ax.grid(alpha=.15)
    if legend:
        ax.legend(loc='upper left',fontsize=7,framealpha=.9,markerscale=1.7)


def draw_three_feature_cloud(ax, config, step, *, legend=False):
    classes=config['n_classes']
    d=config['separations'][step] if 'separations' in config else separation_for_step(step,config['steps'],3)
    x,labels=make_data(0,d,config['n_per_class'],n_classes=classes,n_features=3)
    colors,_=class_appearance(classes)
    vertices=np.array(list(product([-1.,1.],repeat=3)))
    for j,(center,color,marker) in enumerate(zip(generating_centers(d,classes),colors,['o','s','^','D','P'])):
        subset=x[labels==j]
        ax.scatter(*subset.T,s=9,marker=marker,c=color,alpha=.38,edgecolors='none',label=f'Classe {j+1}',depthshade=False)
        for first,second in combinations(vertices,2):
            if np.count_nonzero(first!=second)==1:
                edge=np.stack([first,second])+center
                ax.plot(*edge.T,color=color,linewidth=.45,alpha=.25)
        ax.scatter([center],[center],[center],marker='x',color=color,s=30,linewidths=1.2)
    bound=classes+.1
    ax.set(xlim=(-bound,bound),ylim=(-bound,bound),zlim=(-bound,bound),
           xlabel='Feature 1',ylabel='Feature 2',zlabel='Feature 3',
           title=f'Passo {step}: sovrapposizione {100*(1-step/config["steps"]):.0f}%')
    ax.set_box_aspect((1,1,1))
    ax.view_init(elev=20,azim=-60)
    ax.tick_params(labelsize=7,pad=0)
    ax.xaxis.labelpad=0
    ax.yaxis.labelpad=0
    ax.zaxis.labelpad=0
    if legend:
        ax.legend(loc='upper left',fontsize=6.5,framealpha=.9,markerscale=1.5)


def draw(directory, output):
    config = json.loads((directory/'provenance.json').read_text())['configuration']
    steps = config['steps']
    n_classes = config.get('n_classes',2)
    n_features = config.get('n_features',1)
    prefix = f'overlap_{n_features}features_{n_classes}classes' if n_features>1 else ('overlap' if n_classes==2 else f'overlap_{n_classes}classes')
    class_colors, class_styles = class_appearance(n_classes)
    fixed = f'k{n_classes}'
    limit = n_classes+.1
    examples = []
    for step in range(steps+1):
        d = separation_for_step(step, steps, n_features)
        observations, labels = make_data(0,d,config['n_per_class'],n_classes=n_classes,n_features=n_features)
        posterior = posterior_probabilities(observations[:,0] if n_features==1 else observations,d,n_classes,n_features)
        record={'step':step,'overlap_pct':100*(1-step/steps),'class':labels}
        if n_features==1:
            record['x']=observations[:,0]
        else:
            record.update({f'x{j+1}':observations[:,j] for j in range(n_features)})
        for j in ([1] if n_classes==2 and n_features==1 else range(n_classes)):
            record[f'posterior_class_{j}']=posterior[:,j]
        examples.append(pd.DataFrame(record))
    pd.concat(examples).to_csv(directory/'example_datasets.csv',index=False,float_format='%.12g')
    x = np.linspace(-limit, limit, 2001)
    rows = int(np.ceil((steps+2)/4))
    fig, axes = plt.subplots(rows,4,figsize=(16 if n_features==3 else 15,(3.8 if n_features==3 else 3.5 if n_features==2 else 2.86)*rows),layout='constrained',sharex=True,sharey=True,squeeze=False,subplot_kw={'projection':'3d'} if n_features==3 else {})
    if n_features==3:
        # Matplotlib's automatic layout underestimates projected 3D labels.
        fig.set_layout_engine(None)
        fig.subplots_adjust(left=.02,right=.95,bottom=.09,top=.80 if rows==1 else .90,
                            wspace=.13,hspace=.30)
    for step, ax in enumerate(axes.flat):
        if step > steps:
            ax.axis('off')
            if step == steps+1:
                explanation=(f'{n_features} informative features\nIndependent noise within class\nJoint density overlap calibrated\n\nPoints: replicate 0 observations\nBoxes: class support limits\nCrosses: generating centers' if n_features>1 else f'One observed variable\nEqual class priors\nTriangular class densities\n\nShaded area = overlap\nStep 0: indistinguishable classes\nStep {steps}: exclusive almost surely')
                (ax.text2D if n_features==3 else ax.text)(.03,.88,explanation,transform=ax.transAxes,va='top',fontsize=11,linespacing=1.5)
            continue
        if n_features==3:
            draw_three_feature_cloud(ax,config,step,legend=step==0)
            continue
        if n_features==2:
            draw_two_feature_cloud(ax,config,step,legend=step==0)
            continue
        d = separation_for_step(step, steps)
        curves=[density(x,center) for center in generating_centers(d,n_classes)]
        for j in range(n_classes-1):
            ax.fill_between(x,np.minimum(curves[j],curves[j+1]),color='#aa91b8',alpha=.3,label='Adjacent overlap' if j==0 else None)
        for j,(curve,color,style) in enumerate(zip(curves,class_colors,class_styles)):
            ax.plot(x,curve,color=color,lw=1.8,label=f'Class {j}',linestyle=style)
        ax.set(title=f'Step {step} · adjacent overlap {100*(1-step/steps):.0f}%', xlim=(-limit,limit), ylim=(0,1.08))
        ax.grid(alpha=.15)
        if step % 4 == 0: ax.set_ylabel('Class density')
        if step >= (rows-1)*4: ax.set_xlabel('Observed variable x')
        if step == 0: ax.legend(fontsize=8, loc='lower left')
    fig.suptitle(f'Supplementary analysis · {n_classes} classes · {n_features} feature(s) · {steps+1} calibrated overlap levels',fontsize=18,fontweight='bold')
    save(fig, output/f'{prefix}_distributions')

    summary = pd.read_csv(directory/'sweep_summary.csv')
    perf = pd.read_csv(directory/f'{fixed}_performance.csv')
    fig, axes = plt.subplots(2,3,figsize=(15,8.7),layout='constrained')
    for idx, (metric, label, color) in enumerate(zip(INDEXES, LABELS, COLORS)):
        a = summary[summary.metric == metric].sort_values('step')
        ax = axes[0,idx]
        ax.plot(a.step, a[f'score_{fixed}_mean'], 'o-', color=color, ms=4)
        ax.fill_between(a.step, a[f'score_{fixed}_p10'], a[f'score_{fixed}_p90'], color=color, alpha=.16)
        ax.set(title=f'{label} · fixed K={n_classes}', ylabel='Index value (own scale)')
        axes[1,1].plot(a.step, a[f'selected_{fixed}_rate'], 'o-', color=color, label=label, ms=4)
    for method, color, linestyle in [('KMeans',COLORS[0],'-'),('FCM',COLORS[2],'--')]:
        a = perf[perf.method == method].sort_values('step')
        for ax, column in [(axes[1,0],'test_accuracy'),(axes[1,2],'posterior_MSE')]:
            ax.plot(a.step,a[column+'_mean'],marker='o',linestyle=linestyle,color=color,label=method,ms=4)
            ax.fill_between(a.step,a[column+'_p10'],a[column+'_p90'],color=color,alpha=.13)
    a = perf[perf.method == 'FCM'].sort_values('step')
    axes[1,0].plot(a.step,a.bayes_accuracy,'k:',lw=1.8,label='Bayes reference')
    axes[1,0].set(title=f'Class recovery · held-out data, K={n_classes}',ylabel='Accuracy',ylim=(1/n_classes-.05,1.03))
    axes[1,1].set(title='K selection · minimize mean index',ylabel=f'Fraction selecting K={n_classes}',ylim=(-.03,1.05))
    axes[1,2].set(title=f'Membership vs known posterior · K={n_classes}',ylabel='Mean squared error per class',ylim=(-.01,.27))
    for ax in axes.flat:
        ax.set_xlabel('Separation step (0–10)')
        ax.set_xticks(range(steps+1))
        ax.grid(alpha=.2)
    for ax in axes[1]: ax.legend(fontsize=8,loc='best')
    fig.suptitle(f'Supplementary overlap sweep · {n_features} feature(s) · {config["replicates"]} datasets × {config["initializations"]} initializations per K\n100%, 90%, …, 0% joint overlap · bands: 10th–90th percentiles across datasets',fontsize=16,fontweight='bold')
    save(fig, output/f'{prefix}_results')

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
    fig.suptitle(f'Cluster-count selection · {n_classes} generating classes · {n_features} feature(s)\nK=1 is undefined for all three indices; step 0 has a single observable distribution',fontsize=15,fontweight='bold')
    save(fig,output/f'{prefix}_selected_k')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=Path('results/supplementary_overlap'))
    parser.add_argument('--output-dir',type=Path,default=Path('results/figures'))
    args = parser.parse_args()
    draw(args.input_dir,args.output_dir)
