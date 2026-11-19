"""Build the hundred-feature technical note, README and optional local report."""
import argparse
import html
import json
from pathlib import Path

import pandas as pd

from supplementary_overlap import ROOT, INDEXES

LABELS=['DB / k-means','DB / hardened FCM','SDB / FCM']
URL='https://github.com/albertoarturovergani/Soft-Davies-Bouldin/blob/main/docs/TECHNICAL_NOTES_HUNDRED_FEATURES.md'


def folder(c,d=100):
    suffix=f'_{d}features_{c}classes' if d>1 else (f'_{c}classes' if c>2 else '')
    return ROOT/'results'/('supplementary_overlap'+suffix)


def counts(c,step,d=100,name='selected_k.csv'):
    path=folder(c,d)/name
    if not path.exists():
        return None
    frame=pd.read_csv(path)
    return [int(((frame.step==step)&(frame.metric==m)&(frame.selected_k==c)).sum()) for m in INDEXES]


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join([':--']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])+'\n'


def htable(headers,rows):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'


def read_json(c,name):
    return json.loads((folder(c)/name).read_text())


def bundle(c):
    cv=read_json(c,'convergence_sensitivity.json')
    dg=read_json(c,'diagnostics.json')
    v=counts(c,10)
    convergence=(f"Extending {cv['rerun_configurations']:,} capped FCM fits to 10,000 iterations leaves {cv['still_at_limit']} capped and changes {cv['changed_k_choices']}/660 K choices, including {cv['changed_sdb_choices']}/220 SDB choices. Refits reconstruct the exact shifts from provenance JSON. Primary files retain the 1,000-iteration protocol."
                 if cv['rerun_configurations'] else
                 'No FCM fit reached the 1,000-iteration cap, so no 10,000-iteration refit was needed. The convergence files record this zero-refit audit; they are not an independent longer-iteration test.')
    content=f'''# One hundred features, {c} classes — supplementary analysis

One hundred informative triangular coordinates, independent noise within class, equal priors and collinear centers. Eleven calibrated joint-overlap levels (100%, 90%, …, 0%), 20 independent datasets and 100 starts per K=2–6 give **110,000 configurations**. Each class supplies 100 training and 2,000 held-out samples. All new plot axes are linear.

At disjointness, the mean-index rules recover K={c} in **{v[0]}/20 for DB/k-means**, **{v[1]}/20 for DB/hardened FCM** and **{v[2]}/20 for SDB/FCM**. See [Technical note 7](../../docs/TECHNICAL_NOTES_HUNDRED_FEATURES.md) for the full sweep and comparison with earlier feature counts.

The original FCM/SDB implementation is unchanged. There are {dg['near_coincident_fcm_fits']:,} nearly coincident FCM fits, and undefined-score counts are {dg['undefined_indices']}. Undefined scores remain +inf and are included in every 100-start mean; selected_k=0 means no eligible finite-mean candidate. Distinct-center screening is a separate diagnostic, not the primary rule.

{convergence}

The bundle contains the full synthetic score grid, mean-index choices, held-out performance, examples with all 100 coordinates, overlap calibration and independent numerical checks, degeneracy diagnostics, convergence sensitivity and source/data hashes. No participant or fMRI data are included.
'''
    (folder(c)/'README.md').write_text(content,encoding='utf8')


def main(local_report_dir=None):
    for c in range(2,6):
        bundle(c)
    headers=['Classes',*LABELS]
    endpoint=[[c,*[f'{v}/20' for v in counts(c,10)]] for c in range(2,6)]
    near=[[c,*[f'{v}/20' for v in counts(c,9)]] for c in range(2,6)]
    distinct=[[c,*[f'{v}/20' for v in counts(c,10,name='distinct_clusters_selected_k.csv')]] for c in range(2,6)]
    diagnostics={c:read_json(c,'diagnostics.json') for c in range(2,6)}
    checks={c:read_json(c,'convergence_sensitivity.json') for c in range(2,6)}
    cap=sum(v['rerun_configurations'] for v in checks.values())
    left=sum(v['still_at_limit'] for v in checks.values())
    changed=sum(v['changed_k_choices'] for v in checks.values())
    sdb_changed=sum(v['changed_sdb_choices'] for v in checks.values())
    convergence=(f'Extending all **{cap:,} capped FCM fits** to 10,000 iterations leaves **{left}** at the cap and changes **{changed}/2,640 K choices**, including **{sdb_changed}/880 SDB choices**. Primary results retain the 1,000-iteration protocol.'
                 if cap else
                 '**None of the 440,000 primary FCM fits reached the 1,000-iteration cap**, so no 10,000-iteration refit was needed. The zero-refit audit is not an independent longer-iteration test. Membership convergence does not imply distinct centers or a correct K.')
    calibrated=pd.read_csv(folder(2)/'overlap_calibration.csv')
    metadata=read_json(2,'calibration.json')
    endpoint_choices=[]
    accuracy=[]
    for c in range(2,6):
        selection=pd.read_csv(folder(c)/'selected_k.csv')
        chosen=selection[(selection.step==10)&(selection.metric=='SDB_fcm')].selected_k.value_counts()
        endpoint_choices.append([c,*[int(chosen.get(k,0)) for k in [0,2,3,4,5,6]]])
        performance=pd.read_csv(folder(c)/f'k{c}_performance.csv')
        for step in [9,10]:
            part=performance[performance.step==step].set_index('method')
            accuracy.append([c,f'{100-step*10}%',*[f'{part.loc[m,"test_accuracy_mean"]:.2%}' for m in ['KMeans','FCM']],f'{part.bayes_accuracy.iloc[0]:.2%}'])
    compare=[]
    for c in range(2,6):
        for d in [1,2,3,10,100]:
            v=counts(c,10,d)
            compare.append([c,d,*([f'{n}/20' for n in v] if v is not None else ['Not available']*3)])
    diagnostic_rows=[[c,diagnostics[c]['near_coincident_fcm_fits'],diagnostics[c]['fcm_hard_cluster_count_below_requested'],diagnostics[c]['undefined_indices']['SDB_fcm'],checks[c]['rerun_configurations'],checks[c]['still_at_limit'],checks[c]['changed_sdb_choices']] for c in range(2,6)]
    changed_rates=[]
    for c in range(2,6):
        rates=pd.read_csv(folder(c)/'convergence_recovery_rates.csv')
        for r in rates[rates.changed].itertuples():
            changed_rates.append([c,f'{100-10*r.step}%',LABELS[INDEXES.index(r.metric)],f'{round(20*r.primary_correct_rate)}/20',f'{round(20*r.extended_correct_rate)}/20'])
    rate_table=table(['Classes','Joint overlap','Criterion','Primary recovery','Extended recovery'],changed_rates) if changed_rates else 'No correct-count frequency changes at any sampled level.\n'
    endpoint_summary='; '.join(f'{c} classes: {counts(c,10)[2]}/20' for c in range(2,6))
    binary_comparison=(f"At 10% remaining overlap, binary SDB recovery changes from **{counts(2,9,10)[2]}/20 with ten features** to **{counts(2,9)[2]}/20 with one hundred features** under the primary mean-index rule. The hundred-feature count becomes **{counts(2,9,name='distinct_clusters_selected_k.csv')[2]}/20** after distinct-center screening. Nominal K recovery in a nearly collapsed solution should not be presented as a well-separated clustering.")
    all_interior_screened_out=True
    for c in range(2,6):
        screened=pd.read_csv(folder(c)/'distinct_clusters_selected_k.csv')
        interior=screened[screened.step.between(1,9)&screened.metric.isin(['DB_fcm_hard','SDB_fcm'])]
        all_interior_screened_out &= bool(interior.selected_k.eq(0).all())
    interior_message=('Every FCM candidate fails the conservative distinct-center screen at all nine interior overlap levels in all four hundred-feature studies.' if all_interior_screened_out else 'Per-level distinct-center screening is reported separately below.')
    note=f'''# Technical note 7 — one hundred features across the overlap continuum

**19 November 2026. Supplementary synthetic analysis outside the 2018 paper. All new plot axes are linear.**

The protocol retains **2–5 classes, 11 nominal joint-overlap levels, 20 independent datasets and 100 initializations per candidate K=2–6**. The extension adds **440,000 configurations and 880,000 primary algorithm fits**. All 100 coordinates are informative, with independent triangular noise within class and class centers on the same diagonal line. These are count-selection frequencies across datasets, not classification accuracy or success rates across the 100 starts.

## Main result

For SDB at complete separation: **{endpoint_summary}**.

Correct generating-K selections with **completely disjoint class supports**:

{table(headers,endpoint)}
At **10% remaining joint overlap**:

{table(headers,near)}
{binary_comparison}

{interior_message}

![Hundred-feature recovery from two to five classes](../results/figures/overlap_100features_2_to_5_recovery.png)

The primary curves select the requested K by its arithmetic mean index. In high dimensions, nearly coincident FCM centers can still give numerical hard labels, or a requested K that happens to equal the generating count. Neither alone establishes distinct recovered classes. The separate screening and held-out diagnostics below are essential to interpretation.

SDB's selected K at the disjoint endpoint (number of datasets):

{table(['Classes','No selection','K=2','K=3','K=4','K=5','K=6'],endpoint_choices)}
Held-out classification accuracy with **K fixed to the generating count**, using the best native-objective partition among 100 starts:

{table(['Classes','Joint overlap','K-means accuracy','FCM accuracy','Bayes reference'],accuracy)}
This fixed-K analysis separates the algorithm's ability to assign observations from the index's ranking of candidate K values. In degenerate FCM solutions, argmax labels can depend on tiny departures from uniform membership; their numerical accuracy is not evidence of well-separated fuzzy centers.

## Protocol

Class j has center `(j-(C-1)/2) * d * (1,...,1)` and 100 independent `Triangular(-1,0,1)` noise coordinates, with equal priors. Each class contributes 100 training and 2,000 independent held-out observations. The first ten coordinates and observation ordering exactly match the earlier ten-feature random streams at the same coordinate shift. Base draws are paired across overlap levels; training and test streams are separate.

Candidate K is 2–6. Each dataset and K uses seeds 0–99. The original FCM and SDB implementation files are unchanged: FCM m=2, maximum 1,000 iterations and membership tolerance 1e-7; published SDB q=1, Euclidean p=2 and raw mean memberships. K-means uses one start per seed, maximum 1,000 iterations and tolerance 1e-7. Selection minimizes the arithmetic mean index over all 100 starts; smaller K breaks ties.

As in the ten-feature study, mathematically undefined scores are recorded as +inf with their reason and are never dropped from the mean. Only finite means are eligible. `selected_k=0` means no selection, counted as failure. DB on hardened FCM labels can evaluate fewer occupied groups than requested, so occupancy is explicitly recorded. New score archives retain 17 significant digits and are reloaded with round-trip float parsing.

At complete overlap there is one observable distribution, despite the latent class labels. All three indices are undefined at K=1; that endpoint is omitted from recovery curves and retained only as a descriptive numerical diagnostic. At d=2 the closed supports touch only at a probability-zero boundary and are disjoint almost surely.

## Calibrating joint overlap in 100 dimensions

The target is the adjacent-pair joint-density overlap `OD(d) = integral min(product t(x_r), product t(x_r-d)) dx`, with `t(x)=max(1-|x|,0)`. It is not the per-coordinate overlap or the overlap of projected scatterplots. FCM's fuzzifier stays fixed at m=2.

A one-dimensional likelihood-ratio distribution makes the 100-dimensional integral tractable. Under the first density and conditional on the common support `[d-1,1]`, define `Y=log(t(X-d)/t(X))`. The common-support probability is `M=1-d²/2` for d<1 and `M=(2-d)²/2` for d≥1. Independence and reflection symmetry imply `OD(d)=2*M**D*P(Y1+...+YD>0)` for 0<d<2.

The ratio is strictly increasing in X. For r=exp(y), its inverse is `x=d/(1-r)-1` when d<1 and r<1-d, `x=1-d/(r-1)` when d<1 and r>1/(1-d), and `x=(r+d-1)/(1+r)` otherwise. The triangular CDF therefore gives exact probability masses in each log-ratio bin. FFT convolution raises the discrete characteristic function to the 100th power; half the central-bin mass approximates the continuous zero boundary.

Calibration uses **2^19 bins over approximately [-64,64]**, with a scalar root tolerance of 1e-10. It is checked with **2^21 bins**, then by doubling the domain at the same refined bin width. Maximum refinement discrepancy from a target is **{100*metadata['maximum_fft_refinement_error']:.8f} percentage points**; maximum widened-domain difference is **{metadata['maximum_domain_difference']:.3g}** in probability units. The prespecified checks are 2e-5 and 1e-8 respectively. Endpoints are exact.

An independent method integrates the last coordinate analytically and the other 99 using conditional importance quadrature: **four scrambled Sobol sequences of 2^18 points each**. The largest discrepancy between their average and refined convolution is **{100*metadata['maximum_qmc_difference']:.6f} percentage points**, within the 0.15-percentage-point cross-check tolerance. Individual estimates and scramble standard errors are saved. Convolution is also tested against the exact one-coordinate formula and independent three-coordinate Gauss quadrature. These are numerical checks, not rigorous error bounds. [The ten-feature note](TECHNICAL_NOTES_TEN_FEATURES.md) describes the independent conditional integration identity.

{table(['Step','Target overlap','Refined convolution','Coordinate shift','Per-coordinate overlap'],[[int(r.step),f'{r.target_joint_overlap:.0%}',f'{r.reference_joint_overlap:.7%}',f'{r.per_feature_shift:.10f}',f'{r.per_feature_overlap:.4%}'] for r in calibrated.itertuples()])}
Equal overlap steps are not equal geometric distances. In particular, the final step changes the per-coordinate shift from about {calibrated.iloc[9].per_feature_shift:.4f} to 2. Connecting lines interpolate between sampled levels; this grid cannot locate transitions below 10% remaining overlap. The equal-prior Bayes-accuracy reference is `1-((C-1)/C)*OD`, up to numerical calibration accuracy.

## Distinct-center and convergence checks

The conservative screening rejects an SDB/FCM candidate if any of its 100 starts has minimum center distance divided by training-data RMS radius below 1e-6. DB/hardened FCM additionally requires exactly K occupied labels in every start. The threshold is a diagnostic choice, not a modification of SDB or a universal criterion.

Correct selections at disjointness after screening:

{table(headers,distinct)}
{table(['Classes','Nearly coincident FCM fits','Hard groups < requested K','Undefined SDB','Capped fits refitted','Still capped','Changed SDB choices'],diagnostic_rows)}
Each row describes 110,000 configurations across all K and overlap levels. These counts are not failure rates of the selected model alone.

{convergence}

{rate_table}
Refits regenerate the same data from exact provenance JSON shifts, seeds and package versions. Cached individual refits and dataset/step checkpoints can resume only with matching inputs and source hashes. Reaching the cap is not convergence, and near-coincidence remains a separate issue even when the membership convergence criterion is satisfied. Extended selections, per-level recovery and post-refit screening are saved separately.

## Comparison with earlier feature counts

Correct-count selections with disjoint supports:

{table(['Classes','Features',*LABELS],compare)}
![One through one hundred feature comparison](../results/figures/overlap_1_vs_2_vs_3_vs_10_vs_100_features.png)

The earlier five-class, three-feature experiment remains interrupted and unavailable; it is never plotted as zero. All fifteen previous score archives remain unchanged. Matching joint overlap controls population classification difficulty in this geometry, while distances, memberships and solver behavior still change with dimension. This balanced, collinear design does not establish a general dimensionality effect or universal superiority of an index. Total sample size also grows with class count, and K=2–6 leaves fewer over-counting alternatives when C increases.

## Results for each class count

'''
    for c in range(2,6):
        levels=[[f'{100-10*s}%',*[f'{v}/20' for v in counts(c,s)],*[f'{v}/20' for v in counts(c,s,name='distinct_clusters_selected_k.csv')]] for s in range(11)]
        note+=f'''### {c} classes

![{c}-class recovery with four example datasets](../results/figures/overlap_100features_{c}classes_recovery_it.png)

Examples project all 100 coordinates onto the known center direction and an orthogonal contrast. X-axis ranges adapt per panel and are labeled; projected overlap is not the calibrated joint overlap.

{table(['Joint overlap',*LABELS,*['Screened '+v for v in LABELS]],levels)}
The 100%-overlap row has no identifiable generating count and is descriptive only.

![{c}-class validity sensitivity](../results/figures/overlap_100features_{c}classes_validity.png)

![{c}-class held-out performance and degeneracy](../results/figures/overlap_100features_{c}classes_performance.png)

[Selected-K distribution](../results/figures/overlap_100features_{c}classes_selected_k.png) · [Full synthetic results](../results/supplementary_overlap_100features_{c}classes/)

'''
    note+='''## Reproduction

From the repository root, after installing the benchmark and development dependencies:

```text
python scripts/python/supplementary_overlap_hundred_features.py --classes 2 3 4 5 --workers 12 --checkpoint-root .local_overlap_checkpoints/hundred
python scripts/python/check_hundred_feature_convergence.py --input-dir results/supplementary_overlap_100features_2classes --workers 4
python scripts/python/plot_overlap_hundred_features.py --input-dir results/supplementary_overlap_100features_2classes
python scripts/python/plot_overlap_hundred_features.py --compare-all
python scripts/python/report_hundred_features.py
pytest -q
```

Repeat the per-class convergence and plotting commands for classes 3, 4 and 5. Calibration JSON retains the exact shifts; checkpoints also record source hashes and the execution environment. Use a separate output/checkpoint directory for changed inputs, source or environment. The four new full score grids are synthetic and public. Private original participant-level archives remain excluded; the extension does not change the earlier fMRI or abstract-reproduction results.
'''
    (ROOT/'docs/TECHNICAL_NOTES_HUNDRED_FEATURES.md').write_text(note,encoding='utf8')
    block=f'''<!-- hundred-feature-overlap:start -->
**Hundred-feature extension (19 November 2026):** 2–5 classes, 11 calibrated joint-overlap levels, 20 datasets and 100 starts per K=2–6 add **440,000 configurations**. All new axes are linear.

Correct generating-K selections with disjoint supports:

{table(headers,endpoint)}
At 10% remaining joint overlap:

{table(headers,near)}
{binary_comparison}

**Numerical K selection must be distinguished from distinct recovered clusters.** The report separately measures nearly coincident FCM centers, occupied hard-label groups, conservative candidate screening and held-out accuracy at fixed K. Undefined scores remain +inf and are retained in every 100-start mean.

{convergence}

![Hundred-feature recovery](results/figures/overlap_100features_2_to_5_recovery.png)

Read [Technical note 7: one hundred features](docs/TECHNICAL_NOTES_HUNDRED_FEATURES.md) for the full sweep, four dataset projections per class count, numerical overlap checks and the [comparison with 1, 2, 3 and 10 features](results/figures/overlap_1_vs_2_vs_3_vs_10_vs_100_features.png). Earlier results remain unchanged; the interrupted 3-feature/5-class study remains unavailable. This one collinear geometry does not establish universal dimensionality benefits.
<!-- hundred-feature-overlap:end -->
'''
    readme_path=ROOT/'README.md'
    readme=readme_path.read_text(encoding='utf8')
    begin,end='<!-- hundred-feature-overlap:start -->','<!-- hundred-feature-overlap:end -->'
    if begin in readme:
        start=readme.index(begin)
        stop=readme.index(end)+len(end)
        readme=readme[:start]+block.rstrip()+readme[stop:]
    else:
        readme=readme.replace('## Testing',block+'\n## Testing')
    readme=readme.replace('all fifteen complete 110,000-configuration supplementary grids','all nineteen complete 110,000-configuration supplementary grids')
    readme=readme.replace('ten-feature degeneracy handling and checkpoint compatibility checks','ten-/hundred-feature degeneracy handling, likelihood-ratio convolution and checkpoint compatibility checks')
    readme_path.write_text(readme,encoding='utf8')
    if local_report_dir:
        local_report_dir=Path(local_report_dir)
        local_report_dir.mkdir(parents=True,exist_ok=True)
        import os
        relative=Path(os.path.relpath(ROOT,local_report_dir)).as_posix()
        css='body{font:17px/1.55 system-ui;max-width:1200px;margin:40px auto;padding:0 24px;color:#173047}h1,h2{color:#10283d}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{border:1px solid #ccd5dc;padding:9px;text-align:left}th{background:#edf4f8}img{max-width:100%;height:auto}a{color:#166099}'
        report='<!doctype html><html lang="it"><meta charset="utf-8"><title>SDB: cento feature</title><style>'+css+'</style><body><h1>Cento feature: confronto completo</h1><p>19 novembre 2026 · 2–5 classi · 11 livelli di sovrapposizione · 20 dataset · 100 inizializzazioni per K=2–6.</p><h2>Classi disgiunte</h2>'+htable(headers,endpoint)+'<h2>Sovrapposizione residua del 10%</h2>'+htable(headers,near)+'<p>Un K numericamente corretto non garantisce centri FCM distinti. Le diagnostiche sotto separano scelta del numero, degenerazione e accuratezza a K fissato.</p>'
        report+=f'<p>Nel caso binario con sovrapposizione residua del 10%, SDB passa da {counts(2,9,10)[2]}/20 successi con 10 feature a {counts(2,9)[2]}/20 con 100 feature. Dopo il controllo sui centri distinti, il risultato a 100 feature è {counts(2,9,name="distinct_clusters_selected_k.csv")[2]}/20.</p>'
        if all_interior_screened_out:
            report+='<p>Per tutte le quattro numerosità di classe, nessun candidato FCM supera il controllo conservativo sui centri distinti ai nove livelli intermedi di sovrapposizione. La convergenza delle membership può quindi accompagnarsi a centri quasi coincidenti.</p>'
        report+='<p>La griglia non risolve le transizioni sotto il 10% di sovrapposizione residua. Tutte le feature sono informative, ma i centri restano allineati: questo esperimento descrive una geometria specifica e non dimostra un vantaggio generale del numero di feature.</p>'
        report+=f'<p><a href="{URL}">Technical note 7 su GitHub</a> · <a href="{relative}/docs/TECHNICAL_NOTES_HUNDRED_FEATURES.md">Nota completa locale</a></p>'
        for name in ['overlap_100features_2_to_5_recovery','overlap_1_vs_2_vs_3_vs_10_vs_100_features']:
            report+=f'<img src="{relative}/results/figures/{name}.png" alt="Confronto del numero di cluster">'
        for c in range(2,6):
            report+=f'<h2>{c} classi</h2>'
            for name in ['recovery_it','validity','performance']:
                report+=f'<img src="{relative}/results/figures/overlap_100features_{c}classes_{name}.png" alt="Risultati con {c} classi: {name}">'
        report+='<h2>Controllo di convergenza</h2><p>'+html.escape(convergence.replace('**',''))+'</p></body></html>'
        (local_report_dir/'supplementary_hundred_features.html').write_text(report,encoding='utf8')
        (local_report_dir/'hundred_features_summary.json').write_text(json.dumps(dict(endpoint=endpoint,overlap10=near,distinct_endpoint=distinct,diagnostics=diagnostics,convergence=checks,calibration=metadata),indent=2)+'\n')
        for name in ['report.html','supplementary_ten_features.html']:
            path=local_report_dir/name
            if path.exists():
                old=path.read_text(encoding='utf8')
                if 'supplementary_hundred_features.html' not in old:
                    old=old.replace('</body>','<p><a href="supplementary_hundred_features.html">Nuovo confronto: cento feature, 2–5 classi</a></p></body>')
                    path.write_text(old,encoding='utf8')
    print('Reports ready: technical note 7, README and local report',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-only',type=int,choices=[2,3,4,5])
    parser.add_argument('--local-report-dir',type=Path)
    args=parser.parse_args()
    bundle(args.bundle_only) if args.bundle_only else main(args.local_report_dir)
