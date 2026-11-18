from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'/'python'))
from overlap_geometry import three_feature_overlap, weighted_triangular_overlap
from supplementary_overlap import (density, generating_centers, joint_overlap, make_data,
                                   posterior_probabilities, separation_for_step,
                                   squared_distances, predict_membership)


def test_analytic_weighted_integral_against_adaptive_quadrature():
    rng=np.random.default_rng(62371)
    for d in np.linspace(0,2,11):
        for a,b in rng.uniform(.01,2,size=(12,2)):
            knots=sorted({d-1.,1.,*[v for v in [0.,d] if d-1.<v<1.]})
            for lo,hi in list(zip(knots[:-1],knots[1:])):
                f0=a*density(lo,0)-b*density(lo,d)
                f1=a*density(hi,0)-b*density(hi,d)
                if f0*f1<0:
                    knots.append(lo+(hi-lo)*f0/(f0-f1))
            expected=quad(lambda z:min(a*density(z,0),b*density(z,d)),d-1,1,
                          points=sorted(set(knots)),epsabs=1e-11)[0] if d<2 else 0.
            assert float(weighted_triangular_overlap(a,b,d))==pytest.approx(expected,abs=1e-11)


def test_three_feature_levels_against_direct_three_dimensional_integration():
    # Independent integration of the original 3D minimum, without analytic
    # integration of a coordinate or reusing the Gauss rule under test.
    unit=qmc.Sobol(d=3,scramble=True,seed=20261118).random_base2(20)
    for step in range(11):
        d=separation_for_step(step,n_features=3)
        target=1-step/10
        assert three_feature_overlap(d,512)==pytest.approx(target,abs=1e-7)
        xyz=d-1+(2-d)*unit
        f=np.prod(density(xyz,0),axis=1)
        g=np.prod(density(xyz,d),axis=1)
        direct=np.minimum(f,g).mean()*(2-d)**3
        assert direct==pytest.approx(target,abs=2e-5)
        assert float(joint_overlap(d,3))==pytest.approx(target,abs=1e-10)
        if 0<step<10:
            assert joint_overlap(d,1)>joint_overlap(d,2)>joint_overlap(d,3)


@pytest.mark.parametrize('classes',[2,3,4,5])
def test_three_feature_data_pairing_posteriors_and_bayes_envelope(classes):
    import pandas as pd
    saved=pd.read_csv(Path(__file__).resolve().parents[1]/'results'/f'supplementary_overlap_2features_{classes}classes'/'example_datasets.csv')
    for step in [0,5,10]:
        earlier=saved[saved.step==step]
        d=separation_for_step(step,n_features=2)
        new,labels=make_data(0,d,100,n_classes=classes,n_features=3)
        np.testing.assert_allclose(new[:,:2],earlier[['x1','x2']],atol=1e-10,rtol=1e-10)
        np.testing.assert_array_equal(labels,earlier['class'])
    zero,y=make_data(3,0,3000,n_classes=classes,n_features=3)
    previous,yp=make_data(3,0,3000,n_classes=classes,n_features=2)
    disjoint,yd=make_data(3,2,3000,n_classes=classes,n_features=3)
    test,_=make_data(3,0,3000,test=True,n_classes=classes,n_features=3)
    np.testing.assert_array_equal(zero[:,:2],previous)
    np.testing.assert_array_equal(y,yp)
    np.testing.assert_array_equal(y,yd)
    assert not np.array_equal(zero,test)
    assert zero.shape==(3000*classes,3)
    np.testing.assert_allclose(disjoint-zero,np.repeat(generating_centers(2,classes)[y,None],3,axis=1),atol=1e-15)
    np.testing.assert_allclose(np.cov(zero.T),np.eye(3)/6,atol=.012)
    np.testing.assert_allclose(posterior_probabilities(zero,0,classes,3),1/classes)
    np.testing.assert_allclose(posterior_probabilities(disjoint,2,classes,3),np.eye(classes)[y])
    points=np.random.default_rng(35).uniform(-classes,classes,size=(12000,3))
    for step in range(11):
        centers=generating_centers(separation_for_step(step,n_features=3),classes)
        values=np.prod(density(points[:,None,:],centers[None,:,None]),axis=2)
        envelope=values.sum(axis=1)-np.minimum(values[:,:-1],values[:,1:]).sum(axis=1)
        np.testing.assert_allclose(envelope,values.max(axis=1),atol=1e-12)


def test_third_coordinate_affects_distances_membership_and_density_boundary():
    centers=np.array([[0.,0.,0.],[0.,0.,2.]])
    x=np.array([[0.,0.,0.],[0.,0.,2.],[0.,0.,1.],[2.,0.,0.]])
    np.testing.assert_allclose(squared_distances(x,centers),[[0,4],[4,0],[1,1],[4,8]])
    np.testing.assert_allclose(predict_membership(x,centers),[[1,0],[0,1],[.5,.5],[2/3,1/3]])
    # This point is on x+y+z=3*d/2, but not on the equal-density boundary.
    point=np.array([.85,-.05,-.05])
    assert point.sum()==pytest.approx(3*.5/2)
    assert abs(np.prod(density(point,0))-np.prod(density(point,.5)))>.003
