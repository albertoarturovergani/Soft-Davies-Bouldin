from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.integrate import quad

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'python'))
from supplementary_overlap import (bayes_accuracy, density, generating_centers, make_data,
                                   posterior_probabilities, predict_membership, run_replicate,
                                   separation_for_step, true_posterior, joint_overlap,
                                   squared_distances, evaluate_best)


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


@pytest.mark.parametrize('n_classes',[3,4,5])
def test_multiclass_overlap_and_bayes_reference(n_classes):
    for step in range(11):
        d = separation_for_step(step)
        centers = generating_centers(d,n_classes)
        midpoints = (centers[:-1]+centers[1:])/2
        knots = sorted(set([0,*centers,*(centers-1),*(centers+1),*midpoints]))
        bound = n_classes+1
        for left,right in zip(centers[:-1],centers[1:]):
            overlap = quad(lambda x: min(density(x,left),density(x,right)), -bound,bound,points=knots)[0]
            assert overlap == pytest.approx(1-step/10,abs=1e-10)
        correct = quad(lambda x: np.max(density(x,centers))/n_classes,-bound,bound,points=knots)[0]
        assert correct == pytest.approx(bayes_accuracy(1-step/10,n_classes),abs=1e-10)


@pytest.mark.parametrize('n_classes',[3,4,5])
def test_multiclass_endpoints_and_paired_samples(n_classes):
    x0,y0 = make_data(7,0,100,n_classes=n_classes)
    x2,y2 = make_data(7,2,100,n_classes=n_classes)
    assert x0.shape == (100*n_classes,1)
    np.testing.assert_array_equal(np.bincount(y0),np.full(n_classes,100))
    np.testing.assert_array_equal(y0,y2)
    np.testing.assert_allclose(x2[:,0]-x0[:,0],2*(y0-(n_classes-1)/2))
    np.testing.assert_allclose(posterior_probabilities(x0[:,0],0,n_classes),1/n_classes)
    np.testing.assert_allclose(posterior_probabilities(x2[:,0],2,n_classes),np.eye(n_classes)[y2])


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


@pytest.mark.parametrize('result_name',['supplementary_overlap',*[f'supplementary_overlap_{n}classes' for n in [3,4,5]],*[f'supplementary_overlap_2features_{n}classes' for n in [2,3,4,5]]])
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


def test_two_feature_overlap_by_independent_double_quadrature():
    for step in range(11):
        d=separation_for_step(step,n_features=2)
        if step==10:
            area=0.
        else:
            lo,hi=d-1,1.
            def inner(x):
                points=sorted({v for v in [0,d,d-x] if lo<v<hi})
                return quad(lambda y:min(density(x,0)*density(y,0),density(x,d)*density(y,d)),
                            lo,hi,points=points,epsabs=1e-11)[0]
            area=quad(inner,lo,hi,points=[v for v in [0,d] if lo<v<hi],epsabs=1e-10)[0]
        assert area==pytest.approx(1-step/10,abs=1e-9)
        assert float(joint_overlap(d,2))==pytest.approx(area,abs=1e-9)
        if 0<step<10:
            assert joint_overlap(d,1)>joint_overlap(d,2)


@pytest.mark.parametrize('n_classes',[2,3,4,5])
def test_two_feature_data_posteriors_and_bayes_envelope(n_classes):
    x0,y0=make_data(3,0,2000,n_classes=n_classes,n_features=2)
    x2,y2=make_data(3,2,2000,n_classes=n_classes,n_features=2)
    x1,y1=make_data(3,0,2000,n_classes=n_classes)
    assert x0.shape==(2000*n_classes,2)
    np.testing.assert_array_equal(x0[:,0],x1[:,0])
    np.testing.assert_array_equal(y0,y1)
    np.testing.assert_array_equal(y0,y2)
    offsets=np.repeat(generating_centers(2,n_classes)[y0,None],2,axis=1)
    np.testing.assert_allclose(x2-x0,offsets,atol=1e-15)
    assert np.linalg.det(np.cov(x0.T))>.02
    assert abs(np.corrcoef(x0.T)[0,1])<.05
    np.testing.assert_allclose(posterior_probabilities(x0,0,n_classes,2),1/n_classes)
    np.testing.assert_allclose(posterior_probabilities(x2,2,n_classes,2),np.eye(n_classes)[y2])
    centers=np.repeat(generating_centers(2,n_classes)[:,None],2,axis=1)
    assert evaluate_best(x2,y2,centers,'FCM',2,n_classes,2)['test_accuracy']==1.
    grid=np.stack(np.meshgrid(np.linspace(-n_classes,n_classes,121),np.linspace(-n_classes,n_classes,121)),axis=-1).reshape(-1,2)
    for step in range(11):
        d=separation_for_step(step,n_features=2)
        centers=generating_centers(d,n_classes)
        values=np.prod(density(grid[:,None,:],centers[None,:,None]),axis=2)
        envelope=values.sum(axis=1)-np.minimum(values[:,:-1],values[:,1:]).sum(axis=1)
        np.testing.assert_allclose(envelope,values.max(axis=1),atol=1e-12)
        nearest=np.abs(grid.mean(axis=1)[:,None]-centers[None,:]).argmin(axis=1)
        np.testing.assert_allclose(values[np.arange(len(grid)),nearest],values.max(axis=1),atol=1e-12)


def test_two_feature_membership_and_training_distances_use_both_coordinates():
    x=np.array([[0.,0.],[2.,0.],[1.,1.],[0.,2.]])
    centers=np.array([[0.,0.],[2.,0.]])
    np.testing.assert_allclose(squared_distances(x,centers),[[0,4],[4,0],[2,2],[4,8]])
    np.testing.assert_allclose(predict_membership(x,centers),[[1,0],[0,1],[.5,.5],[2/3,1/3]])
