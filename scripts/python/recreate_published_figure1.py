from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

AIM = {
    'datasets':['Cancer','Crab','Glass','Iris','Ovarian','Simple','Thyroid','Wine','RestROIs','RestTOIs','TaskROIs','TaskTOIs'],
    'DB k-means':[0.7628,0.6113,0.9167,0.4048,0.7487,0.4428,0.9802,0.4612,0.4332,1.2417,0.4427,1.2432],
    'DB c-means*':[0.7630,0.6137,1.0658,0.4048,0.7520,0.4375,1.0105,0.4577,0.4519,1.2812,0.4336,1.2830],
    'SDB c-means':[0.6058,0.5802,0.6723,0.5904,0.5553,0.4637,0.7516,0.5524,0.5563,1.0565,0.5618,1.0484],
}
RIM = {
    'datasets':['Cancer','Crab','Glass','Iris','Ovarian','Simple','Thyroid','Wine'],
    'DB k-means':[0.7628,0.6113,1.0054,0.6693,0.7487,0.4428,1.1370,0.5412],
    'DB c-means*':[0.7630,0.6137,1.0752,0.6696,0.7520,0.4375,1.0105,0.5394],
    'SDB c-means':[0.6058,0.6434,0.6779,0.7261,0.6879,0.4637,0.7516,0.6675],
}
MRC = {
    'datasets':list(range(1,11)),
    'DB k-means':[0.0000,0.7564,0.7410,0.8596,0.8572,0.8948,0.8960,0.9064,0.9492,0.9917],
    'DB c-means*':[0.0000,0.7575,0.7492,0.8797,0.9300,1.0860,1.1997,1.2995,1.2069,1.1550],
    'SDB c-means':[0.0000,0.6607,0.7104,0.6889,0.6968,0.8531,0.8306,0.8407,0.7705,0.6712],
}

keys = ['DB k-means','DB c-means*','SDB c-means']
configs = [AIM,RIM,MRC]
headers = ['ABSOLUTE INDEXES MINIMIZATION (AIM)','RELATIVE INDEXES MINIMIZATION (RIM)','MEASURES IN RELATION TO CLUSTERS (MRC)']
subtitles = ['Datasets perspective','Datasets perspective','Overall Datasets Median']
fig = plt.figure(figsize=(19.2,9.1),facecolor='white')
gs = fig.add_gridspec(2,3,top=.91,bottom=.18,left=.058,right=.955,hspace=.72,wspace=.38,height_ratios=[1.6,.83])
styles = [dict(color='black',marker='+',linestyle=':',linewidth=1.1,markersize=8,markeredgewidth=1.1),dict(color='black',marker='*',linestyle=':',linewidth=1.0,markersize=8,markeredgewidth=.7),dict(color='#df3342',marker='D',markerfacecolor='none',linestyle='-',linewidth=1.3,markersize=6,markeredgewidth=1.0)]
for i,(data,header,subtitle) in enumerate(zip(configs,headers,subtitles)):
    ax = fig.add_subplot(gs[0,i])
    X = np.asarray(data['datasets'])
    x = np.arange(len(X)) if i<2 else X
    for k,st in zip(keys,styles):
        ax.plot(x,data[k],label=k,**st)
    ax.set_title(header+'\n'+subtitle,fontweight='bold',fontsize=13.6,linespacing=1.12,pad=8)
    ax.set_ylim(0,1.5)
    ax.set_yticks([.25,.5,.75,1,1.25])
    ax.set_yticklabels(['0.25','0.5','0.75','1','1.25'])
    ax.grid(which='major',color='#bdbdbd',alpha=.50,linewidth=.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():spine.set_color('#3c3c3c')
    ax.tick_params(axis='both',labelsize=11)
    if i<2:
        ax.set_xticks(x)
        ax.set_xticklabels([s.lower() if s.startswith(('Cancer','Crab','Glass','Iris','Ovarian','Simple','Thyroid','Wine')) else s for s in X],rotation=47,ha='right')
        ax.set_xlim(-1,len(X))
    else:
        ax.set_xlim(0,11)
        ax.set_xticks(range(12))
        ax.set_xlabel('Clusters',fontsize=12)
        ax.yaxis.tick_right()
    ax.legend(loc='upper left',fontsize=11,frameon=True,edgecolor='#666666',fancybox=False,framealpha=1,borderpad=.35,labelspacing=.25,handlelength=2.4)
    bx = fig.add_subplot(gs[1,i])
    box = bx.boxplot([data[k] for k in keys],widths=.46,patch_artist=False,showfliers=True,
        boxprops=dict(color='#3255c9',linewidth=1.0),whiskerprops=dict(color='#333333',linewidth=1.0),
        capprops=dict(color='#333333',linewidth=1.0),medianprops=dict(color='#df3342',linewidth=1.2),
        flierprops=dict(marker='+',markersize=8,markeredgecolor='#df3342',markerfacecolor='none'))
    bx.set_title(f'Measures boxplot ({["AIM","RIM","MRC"][i]})',fontsize=13.3,fontweight='bold',pad=8)
    bx.set_xticks([1,2,3]);bx.set_xticklabels(keys,rotation=42,ha='right')
    bx.set_ylim(0,1.45)
    bx.set_yticks([.25,.5,.75,1,1.25])
    bx.set_yticklabels(['0.25','0.5','0.75','1','1.25'])
    bx.grid(axis='y',color='#bdbdbd',alpha=.50,linewidth=.8)
    bx.set_axisbelow(True)
    bx.tick_params(axis='both',labelsize=11)
    for spine in bx.spines.values():spine.set_color('#3c3c3c')
    if i==2:bx.yaxis.tick_right()

out=Path(__file__).resolve().parents[2]/'results'/'figures'
out.mkdir(parents=True,exist_ok=True)
for ext in ['png','svg','pdf']:
    path=out/f'figure1_published_tables_reference.{ext}'
    if ext=='png':fig.savefig(path,dpi=240,facecolor='white')
    else:fig.savefig(path,facecolor='white')
    print(path, path.stat().st_size)
plt.close(fig)
