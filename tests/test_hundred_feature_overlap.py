from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import brentq
from scipy.special import logsumexp
from scipy.stats import triang

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/python'))
from hundred_feature_geometry import log_ratio_cdf, product_overlap
from overlap_geometry import three_feature_overlap
from supplementary_overlap import posterior_probabilities
from supplementary_overlap_ten_features import make_data as ten_data
from supplementary_overlap_hundred_features import make_data, checkpoint_manifest, run_cell, score_means, select_k


def test_log_ratio_distribution_against_independent_scalar_root():
    density=lambda x:max(1-abs(x),0)
    for d in [.03,.6,1.4]:
        for y in [-3.,-.2,0.,.2,3.]:
            x=brentq(lambda x:np.log(density(x-d)/density(x))-y,d-1+1e-12,1-1e-12,xtol=1e-14)
            lower=triang.cdf(d-1,.5,loc=-1,scale=2)
            expected=(triang.cdf(x,.5,loc=-1,scale=2)-lower)/(1-lower)
            assert log_ratio_cdf(y,d)==pytest.approx(expected,abs=2e-12)


def test_convolution_against_analytic_and_independent_three_dimensional_integrals():
    for d in [.03,.2,.8,1.,1.9]:
        assert product_overlap(d,1,power=17,extent=32)==pytest.approx((1-d/2)**2,abs=2e-8)
    for d in [.06,.25,.6,1.]:
        assert product_overlap(d,3,power=17,extent=32)==pytest.approx(three_feature_overlap(d),abs=2e-7)
    assert product_overlap(0.)==1 and product_overlap(2.)==0


def test_hundred_dimensional_convolution_refinement_and_wider_domain():
    previous=1.
    for d in [.008,.025,.055,.11]:
        a=product_overlap(d,power=18,extent=32)
        b=product_overlap(d,power=20,extent=64)
        assert a==pytest.approx(b,abs=2e-6)
        assert 0<a<previous
        previous=a


@pytest.mark.parametrize('classes',[2,3,4,5])
def test_hundred_features_preserve_first_ten_random_streams(classes):
    x,y=make_data(3,0.,600,n_classes=classes)
    old,yo=ten_data(3,0.,600,n_classes=classes)
    assert x.shape==(classes*600,100)
    np.testing.assert_array_equal(x[:,:10],old)
    np.testing.assert_array_equal(y,yo)
    np.testing.assert_allclose(x.var(axis=0),1/6,atol=.022)
    separated,ys=make_data(3,2.,600,n_classes=classes)
    np.testing.assert_array_equal(y,ys)
    np.testing.assert_allclose(separated-x,np.repeat((2*(y-(classes-1)/2))[:,None],100,axis=1),atol=1e-15)
    np.testing.assert_allclose(posterior_probabilities(separated,2.,classes,100),np.eye(classes)[y])
    xt,_=make_data(3,0.,600,n_classes=classes,test=True)
    assert not np.array_equal(x,xt)


def test_hundred_feature_posteriors_against_log_domain_reference():
    classes,d=5,.1083818
    x,_=make_data(1,d,300,n_classes=classes,test=True)
    centers=(np.arange(classes)-(classes-1)/2)*d
    density=np.maximum(1-np.abs(x[:,None,:]-centers[None,:,None]),0)
    with np.errstate(divide='ignore'):
        log_density=np.log(density).sum(axis=2)
    expected=np.exp(log_density-logsumexp(log_density,axis=1,keepdims=True))
    actual=posterior_probabilities(x,d,classes,100)
    np.testing.assert_allclose(actual,expected,atol=2e-14,rtol=2e-13)
    assert np.isfinite(actual).all()


def test_hundred_feature_cell_and_checkpoint_compatibility(tmp_path):
    config=dict(n_classes=2,n_features=100,replicates=1,initializations=2,steps=10,
                n_per_class=15,test_per_class=30,max_k=3,max_iter=1000,tol=1e-7,separations=[0.]*10+[2.])
    rep,step,raw,best=run_cell((config,0,10))
    assert (rep,step)==(0,10) and len(raw)==4 and len(best)==4
    assert all(row['test_accuracy']==1 for row in best if row['k']==2)
    checkpoint_manifest(tmp_path,config)
    checkpoint_manifest(tmp_path,config)
    with pytest.raises(ValueError,match='mismatch'):
        checkpoint_manifest(tmp_path,{**config,'n_features':10})


def test_hundred_feature_refit_uses_exact_saved_dataset_shift():
    from check_hundred_feature_convergence import extend
    config=dict(n_classes=2,n_per_class=30,tol=1e-7,separations=[0.]*10+[2.])
    # The CSV field is deliberately inconsistent: exact checkpoint/provenance
    # shifts must define the regenerated data. At zero separation FCM collapses.
    _,result=extend((0,dict(replicate=0,step=0,k=2,seed=0,separation=2.),config))
    assert result['fcm_near_coincident']


@pytest.mark.parametrize('classes',[2,3,4,5])
def test_saved_hundred_feature_study(classes):
    import json
    directory=Path(__file__).resolve().parents[1]/'results'/f'supplementary_overlap_100features_{classes}classes'
    if not (directory/'diagnostics.json').exists():
        pytest.skip('Hundred-feature study has not been generated yet')
    config=json.loads((directory/'provenance.json').read_text())['configuration']
    assert config['n_features']==100 and config['initializations']==100 and config['replicates']==20
    raw=pd.read_csv(directory/'per_initialization.csv.gz',float_precision='round_trip')
    assert len(raw)==110000 and not raw.duplicated(['replicate','step','k','seed']).any()
    assert (raw.groupby(['replicate','step','k']).size()==100).all()
    for metric in ['DB_kmeans','DB_fcm_hard','SDB_fcm']:
        assert (np.isfinite(raw[metric])==(raw[metric+'_status']=='ok')).all()
        assert not raw[metric].isna().any()
    pd.testing.assert_frame_equal(select_k(score_means(raw)),pd.read_csv(directory/'selected_k.csv'))
    pd.testing.assert_frame_equal(select_k(score_means(raw,screened=True)),pd.read_csv(directory/'distinct_clusters_selected_k.csv'))
    calibration=pd.read_csv(directory/'overlap_calibration.csv')
    assert calibration.reference_absolute_error.max()<2e-5
    assert calibration.qmc_reference_absolute_error.max()<.0015
