from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.integrate import quad

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'python'))
from supplementary_overlap import (bayes_accuracy, density, generating_centers, make_data,
                                   posterior_probabilities, predict_membership, run_replicate,
                                   separation_for_step, true_posterior)


def test_exact_ten_percent_overlap_steps():
    for step in range(11):
        d = separation_for_step(step)
        knots = sorted(set([-1-d/2, -d/2, 1-d/2, -1+d/2, d/2, 1+d/2, 0]))
        area = quad(lambda x: min(density(x, -d/2), density(x, d/2)), -3, 3, points=knots)[0]
        assert area == pytest.approx(1-step/10, abs=1e-10)


def test_overlap_endpoints_and_posterior():
    for step in [0, 10]:
        d = separation_for_step(step)
        x, y = make_data(0, d, 1000)
        p = true_posterior(x[:, 0], d)
        np.testing.assert_allclose(p, .5 if step == 0 else y)
    x = np.array([[-1.], [0.], [1.]])
    np.testing.assert_allclose(predict_membership(x, np.array([[-1.], [1.]])), [[1, 0], [.5, .5], [0, 1]])


def test_paired_sweep_and_independent_test_sample():
    x0, y0 = make_data(4, 0)
    x1, y1 = make_data(4, 1.5)
    np.testing.assert_array_equal(y0, y1)
    np.testing.assert_allclose(x1[:, 0]-x0[:, 0], (y0-.5)*1.5)
    xt, _ = make_data(4, 0, test=True)
    assert not np.array_equal(x0, xt)


def test_three_class_overlap_and_bayes_reference():
    for step in range(11):
        d = separation_for_step(step)
        centers = generating_centers(d,3)
        knots = sorted(set([0,*centers,*(centers-1),*(centers+1),-d/2,d/2]))
        for left,right in zip(centers[:-1],centers[1:]):
            overlap = quad(lambda x: min(density(x,left),density(x,right)), -4,4,points=knots)[0]
            assert overlap == pytest.approx(1-step/10,abs=1e-10)
        correct = quad(lambda x: np.max(density(x,centers))/3,-4,4,points=knots)[0]
        assert correct == pytest.approx(bayes_accuracy(1-step/10,3),abs=1e-10)


def test_three_class_endpoints_and_paired_samples():
    x0,y0 = make_data(7,0,100,n_classes=3)
    x2,y2 = make_data(7,2,100,n_classes=3)
    assert x0.shape == (300,1)
    np.testing.assert_array_equal(np.bincount(y0),[100,100,100])
    np.testing.assert_array_equal(y0,y2)
    np.testing.assert_allclose(x2[:,0]-x0[:,0],2*(y0-1))
    np.testing.assert_allclose(posterior_probabilities(x0[:,0],0,3),1/3)
    np.testing.assert_allclose(posterior_probabilities(x2[:,0],2,3),np.eye(3)[y2])


def test_binary_seeded_scores_remain_reproducible():
    import json
    import pandas as pd
    directory=Path(__file__).resolve().parents[1]/'results'/'supplementary_overlap'
    config=json.loads((directory/'provenance.json').read_text())['configuration']
    config.update(initializations=1,max_k=2)
    rows,_=run_replicate(config,0)
    actual=pd.DataFrame(rows).set_index('step').sort_index()
    saved=pd.read_csv(directory/'per_initialization.csv.gz')
    saved=saved[(saved.replicate==0)&(saved.k==2)&(saved.seed==0)].set_index('step').sort_index()
    columns=['DB_kmeans','DB_fcm_hard','SDB_fcm','objective_kmeans','objective_fcm']
    np.testing.assert_allclose(actual[columns],saved[columns],rtol=1e-12,atol=1e-12)


@pytest.mark.parametrize('result_name',['supplementary_overlap','supplementary_overlap_3classes'])
def test_saved_supplementary_grid(result_name):
    import json
    import pandas as pd
    directory = Path(__file__).resolve().parents[1] / 'results' / result_name
    if not (directory / 'per_initialization.csv.gz').exists():
        pytest.skip('Full supplementary experiment has not been generated yet')
    config = json.loads((directory / 'provenance.json').read_text())['configuration']
    raw = pd.read_csv(directory / 'per_initialization.csv.gz')
    assert config['steps'] >= 10 and config['initializations'] == 100
    assert len(raw) == config['replicates'] * (config['steps']+1) * (config['max_k']-1) * 100
    assert not raw.duplicated(['replicate', 'step', 'k', 'seed']).any()
    assert set(raw.replicate)==set(range(config['replicates']))
    expected = {(step, k, seed) for step in range(config['steps']+1) for k in range(2,config['max_k']+1) for seed in range(100)}
    for _, group in raw.groupby('replicate'):
        assert set(zip(group.step, group.k, group.seed)) == expected
    columns = ['DB_kmeans', 'DB_fcm_hard', 'SDB_fcm']
    assert np.isfinite(raw[columns]).all().all()
    saved = pd.read_csv(directory/'mean_scores.csv').set_index(['replicate','step','k']).sort_index()
    actual = raw.groupby(['replicate','step','k'])[columns].mean().sort_index()
    np.testing.assert_allclose(saved[columns], actual[columns], rtol=1e-10)
    selection = pd.read_csv(directory/'selected_k.csv')
    for (replicate, step, metric), group in selection.groupby(['replicate','step','metric']):
        assert len(group) == 1
        assert group.selected_k.iloc[0] == actual.loc[(replicate,step),metric].idxmin()
