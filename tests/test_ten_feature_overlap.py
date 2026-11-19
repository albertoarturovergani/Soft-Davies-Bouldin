from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'/'python'))
from overlap_geometry import three_feature_overlap
from supplementary_overlap import make_data as earlier_data, posterior_probabilities
from supplementary_overlap_ten_features import (checkpoint_manifest, fcm_scores, make_data,
                                                run_cell, score_means, score_or_infinity, select_k)
from sdb import soft_davies_bouldin
from ten_feature_geometry import conditional_overlap


def test_conditional_integration_against_independent_low_dimensional_rule():
    unit = qmc.Sobol(d=2, scramble=True, seed=77143).random_base2(18)
    for d in [.06,.25,.6,1.,1.5]:
        assert conditional_overlap(d,unit) == pytest.approx(three_feature_overlap(d), abs=2e-5)
    assert conditional_overlap(0.,unit) == 1
    assert conditional_overlap(2.,unit) == 0


def test_ten_dimensional_overlap_against_direct_joint_density_integral():
    # Independent estimator: all ten coordinates from the untruncated first
    # density, and min(1, q/p); no weighted analytic conditional integral.
    unit = qmc.Sobol(d=10, scramble=True, seed=712219).random_base2(20)
    x = np.where(unit < .5, np.sqrt(2*unit)-1, 1-np.sqrt(2*(1-unit)))
    p = np.maximum(1-np.abs(x),0)
    reference = qmc.Sobol(d=9,scramble=True,seed=713001).random_base2(19)
    previous = 1.
    for d in [.03,.12,.25,.40,.8]:
        q = np.maximum(1-np.abs(x-d),0)
        direct = np.minimum(1,np.prod(q/p,axis=1)).mean()
        result = conditional_overlap(d,reference)
        assert result == pytest.approx(direct,abs=4e-4)
        assert 0 <= result < previous
        previous = result


@pytest.mark.parametrize('classes',[2,3,4,5])
def test_paired_ten_feature_data_and_endpoints(classes):
    x,y = make_data(3,0.,3000,n_classes=classes)
    old,yo = earlier_data(3,0.,3000,n_classes=classes,n_features=3)
    np.testing.assert_array_equal(x[:,:3],old)
    np.testing.assert_array_equal(y,yo)
    np.testing.assert_allclose(np.cov(x.T),np.eye(10)/6,atol=.012)
    x2,y2 = make_data(3,2.,3000,n_classes=classes)
    np.testing.assert_array_equal(y,y2)
    np.testing.assert_allclose(x2-x,np.repeat((2*(y-(classes-1)/2))[:,None],10,axis=1),atol=1e-15)
    np.testing.assert_allclose(posterior_probabilities(x,0.,classes,10),1/classes)
    np.testing.assert_allclose(posterior_probabilities(x2,2.,classes,10),np.eye(classes)[y])
    xt,_ = make_data(3,0.,3000,n_classes=classes,test=True)
    assert not np.array_equal(x,xt)


def test_undefined_index_policy_and_degenerate_candidate_screen():
    x = np.array([[-1.]*10,[1.]*10])
    c = np.zeros((2,10))
    u = np.full((2,2),.5)
    score, reason = score_or_infinity(soft_davies_bouldin,x,c,u)
    assert score == np.inf and 'coincide' in reason
    diagnostics = fcm_scores(x,c,u)
    assert diagnostics['fcm_near_coincident'] and diagnostics['fcm_hard_clusters'] == 1
    rows = []
    for k, values in [(2,[1.,1.]),(3,[.2,np.inf])]:
        for seed,value in enumerate(values):
            rows.append(dict(replicate=0,step=1,k=k,seed=seed,DB_kmeans=value,
                             DB_fcm_hard=value,SDB_fcm=value,fcm_near_coincident=True,fcm_hard_clusters=1))
    raw = pd.DataFrame(rows)
    assert (select_k(score_means(raw)).selected_k == 2).all()
    screened = select_k(score_means(raw,screened=True)).set_index('metric')
    assert screened.loc['SDB_fcm','selected_k'] == 0
    assert screened.loc['DB_fcm_hard','selected_k'] == 0
    assert screened.loc['DB_kmeans','selected_k'] == 2
    with pytest.raises(ValueError,match='Expected'):
        score_or_infinity(soft_davies_bouldin,x,c,np.ones((3,2)))


def test_cell_execution_and_checkpoint_guard(tmp_path):
    config = dict(n_classes=2,n_features=10,replicates=1,initializations=2,steps=10,
                  n_per_class=15,test_per_class=30,max_k=3,max_iter=1000,tol=1e-7,
                  separations=[0.]*10+[2.])
    rep,step,raw,best = run_cell((config,0,10))
    assert (rep,step) == (0,10) and len(raw) == 4 and len(best) == 4
    assert all(row['test_accuracy'] == 1 for row in best if row['k'] == 2)
    checkpoint_manifest(tmp_path,config)
    checkpoint_manifest(tmp_path,config)
    with pytest.raises(ValueError,match='mismatch'):
        checkpoint_manifest(tmp_path,{**config,'initializations':3})


@pytest.mark.parametrize('classes',[2,3,4,5])
def test_saved_ten_feature_study(classes):
    import json
    directory = Path(__file__).resolve().parents[1]/'results'/f'supplementary_overlap_10features_{classes}classes'
    if not (directory/'diagnostics.json').exists():
        pytest.skip('Ten-feature primary study has not been generated yet')
    config = json.loads((directory/'provenance.json').read_text())['configuration']
    assert config['initializations'] == 100 and config['replicates'] == 20 and config['steps'] == 10
    raw = pd.read_csv(directory/'per_initialization.csv.gz')
    assert len(raw) == 110000 and not raw.duplicated(['replicate','step','k','seed']).any()
    assert (raw.groupby(['replicate','step','k']).size() == 100).all()
    for metric in ['DB_kmeans','DB_fcm_hard','SDB_fcm']:
        assert (np.isfinite(raw[metric]) == (raw[metric+'_status'] == 'ok')).all()
        assert not raw[metric].isna().any()
    pd.testing.assert_frame_equal(select_k(score_means(raw)),pd.read_csv(directory/'selected_k.csv'))
    pd.testing.assert_frame_equal(select_k(score_means(raw,screened=True)),pd.read_csv(directory/'distinct_clusters_selected_k.csv'))
    calibration = pd.read_csv(directory/'overlap_calibration.csv')
    assert calibration.reference_absolute_error.max() < 2e-4
