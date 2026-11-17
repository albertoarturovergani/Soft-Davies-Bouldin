from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'/'python'))
from summarize_abstract import load_scores,aggregate,report,two_way_anova

ROOT=Path(__file__).resolve().parents[1]
DIR=ROOT/'results'/'abstract_reproduction'
FILES=[DIR/'ground_100seeds_k2-10_scores.csv.gz',DIR/'fmri_10seeds_k2-10_scores.csv.gz']

def test_2018_abstract_counts_from_recomputed_scores():
    df=load_scores(DIR,FILES,100,10)
    assert len(df)==13320
    avg=aggregate(df,ground_runs=100,fmri_runs=10)
    aim,rim,mrc=report(avg)
    assert int(aim.SDB_below_both.sum())==7
    assert int(rim.SDB_below_both.sum())==4
    assert int(mrc.SDB_below_both.sum())==9
    true_k={'Cancer':2,'Crab':2,'Glass':2,'Iris':3,'Ovarian':2,'Simple':4,'Thyroid':3,'Wine':3}
    assert int(sum(row['best_k_SDB_cmeans']==true_k.get(row['dataset'],-1) for row in aim.to_dict('records')))==3
    assert two_way_anova(aim,'dataset')['p_method']>0.05
    assert two_way_anova(rim,'dataset')['p_method']>0.05


def test_100_seed_public_aggregate_summaries():
    directory = ROOT / 'results' / 'reproduction_100seeds'
    averaged = pd.read_csv(directory / 'aggregated_scores.csv')
    assert len(averaged) == 108 and averaged.dataset.nunique() == 12
    assert not averaged.duplicated(['dataset', 'k']).any()
    assert averaged.replicates.eq(100).all()
    aim, rim, mrc = report(averaged)
    assert tuple(int(table.SDB_below_both.sum()) for table in (aim, rim, mrc)) == (7, 4, 9)
    for name, actual, key in [('AIM', aim, 'dataset'), ('RIM', rim, 'dataset'), ('MRC', mrc, 'k')]:
        saved = pd.read_csv(directory / f'{name}_summary.csv').set_index(key).sort_index()
        actual = actual.set_index(key).sort_index()
        columns = ['DB_kmeans', 'DB_cmeans_hard', 'SDB_cmeans']
        np.testing.assert_allclose(actual[columns], saved[columns], rtol=1e-7, atol=1e-8)


def test_local_100_seed_fmri_grid_when_private_scores_available():
    directory = ROOT / 'results' / 'reproduction_100seeds'
    files = [directory / f'{name}_100seeds_k2-10_scores.csv.gz' for name in ('ground', 'fmri')]
    if not all(path.exists() for path in files):
        pytest.skip('Individual 100-start scores are retained locally; public aggregates are tested separately')
    raw = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    assert len(raw) == 68400
    assert not raw.duplicated(['dataset', 'subject', 'k_tested', 'seed']).any()
    assert raw.error.isna().all()
    columns = ['DB_kmeans', 'DB_cmeans_hard', 'SDB_cmeans']
    assert np.isfinite(raw[columns].to_numpy()).all()
    expected_subjects = {'RestROIs': 19, 'RestTOIs': 19, 'TaskROIs': 15, 'TaskTOIs': 15}
    fmri = raw[raw.dataset.isin(expected_subjects)]
    assert len(fmri) == 61200
    assert fmri.subject.str.fullmatch(r'subject_\d{2}').all()
    expected_grid = {(k, seed) for k in range(2, 11) for seed in range(100)}
    for name, subset in raw.groupby('dataset'):
        assert subset.subject.nunique(dropna=False) == expected_subjects.get(name, 1)
        for _, subject in subset.groupby('subject', dropna=False):
            assert set(zip(subject.k_tested, subject.seed)) == expected_grid
    df = load_scores(directory, files, 100, 100)
    averaged = aggregate(df, ground_runs=100, fmri_runs=100)
    aim, rim, mrc = report(averaged)
    assert tuple(int(table.SDB_below_both.sum()) for table in (aim, rim, mrc)) == (7, 4, 9)
    for name, actual, key in [('aggregated_scores', averaged, ['dataset', 'k']), ('AIM_summary', aim, ['dataset']), ('RIM_summary', rim, ['dataset']), ('MRC_summary', mrc, ['k'])]:
        saved = pd.read_csv(directory / f'{name}.csv')
        actual = actual.set_index(key).sort_index()
        saved = saved.set_index(key).sort_index()
        assert actual.index.equals(saved.index)
        np.testing.assert_allclose(actual[columns], saved[columns], rtol=1e-7, atol=1e-8)
    pilot = df[df.dataset.isin(expected_subjects) & (df.seed < 10) | ~df.dataset.isin(expected_subjects)]
    pilot_aim, _, pilot_mrc = report(aggregate(pilot, ground_runs=100, fmri_runs=10))
    assert aim.loc[aim.dataset.isin(expected_subjects), 'best_k_SDB_cmeans'].tolist() == [4, 2, 2, 2]
    np.testing.assert_array_equal(aim.best_k_SDB_cmeans, pilot_aim.best_k_SDB_cmeans)
    np.testing.assert_allclose(mrc[columns], pilot_mrc[columns], rtol=1e-7, atol=1e-8)
