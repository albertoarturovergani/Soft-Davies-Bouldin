import numpy as np
import pytest
from sdb import soft_davies_bouldin, davies_bouldin_crisp, fuzzy_c_means


def fixture():
    X = np.array([[0.0],[1.0],[4.0],[5.0]])
    centers = np.array([[0.5],[4.5]])
    U = np.array([[1.0,0.0],[1.0,0.0],[0.0,1.0],[0.0,1.0]])
    return X, centers, U


def test_hand_computable_published_sdb():
    X, centers, U = fixture()
    assert soft_davies_bouldin(X, centers, U) == pytest.approx(0.5625)
    assert davies_bouldin_crisp(X, np.argmax(U, axis=1)) == pytest.approx(0.25)


def test_historical_power_is_separate():
    X, centers, U = fixture()
    U = np.array([[0.9,0.1],[0.8,0.2],[0.2,0.8],[0.1,0.9]])
    assert soft_davies_bouldin(X, centers, U) != pytest.approx(soft_davies_bouldin(X, centers, U, membership_power=2))


def test_invariances():
    X, centers, U = fixture()
    ref = soft_davies_bouldin(X, centers, U)
    assert soft_davies_bouldin(X+9, centers+9, U) == pytest.approx(ref)
    assert soft_davies_bouldin(X*3, centers*3, U) == pytest.approx(ref)
    assert soft_davies_bouldin(X, centers[::-1], U[:, ::-1]) == pytest.approx(ref)


def test_q_2_and_finite():
    X, centers, U = fixture()
    assert np.isfinite(soft_davies_bouldin(X, centers, U, q=2))
    assert soft_davies_bouldin(X, centers, U, q=2) > soft_davies_bouldin(X, centers, U)


@pytest.mark.parametrize("patch", [
    {"U": np.ones((4,2))},
    {"U": np.ones((4,1))},
    {"centers": np.array([[0.5],[0.5]])},
    {"X": np.array([[0.0],[np.nan],[4.0],[5.0]])},
])
def test_invalid_input_rejected(patch):
    X, centers, U = fixture()
    data = {"X":X,"centers":centers,"U":U}
    data.update(patch)
    with pytest.raises(ValueError):
        soft_davies_bouldin(data['X'], data['centers'], data['U'])


def test_fuzzy_c_means_properties():
    X = np.array([[0.,0.],[0.,1.],[4.,4.],[5.,5.]])
    centers, U, iters = fuzzy_c_means(X, 2, m=2, seed=1)
    assert centers.shape == (2,2)
    assert U.shape == (4,2)
    assert np.all((U>=0)&(U<=1))
    np.testing.assert_allclose(U.sum(axis=1), 1, atol=1e-12)
    assert 1 <= iters <= 300
    assert np.isfinite(soft_davies_bouldin(X, centers, U))


def test_sklearn_iris_reference():
    sklearn = pytest.importorskip('sklearn')
    from sklearn.datasets import load_iris
    X = load_iris().data
    centers, U, _ = fuzzy_c_means(X, 3, m=2, seed=42)
    assert soft_davies_bouldin(X, centers, U) == pytest.approx(0.7261, abs=0.0001)
