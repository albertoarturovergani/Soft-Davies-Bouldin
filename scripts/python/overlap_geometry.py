"""Joint overlap of diagonal shifts of independent triangular coordinates.

For three features the likelihood boundary is generally curved. Integrate the
minimum of the two joint densities, rather than a plane-based approximation.
"""
from functools import lru_cache

import numpy as np
from scipy.special import roots_legendre


def weighted_triangular_overlap(a, b, separation):
    """Exact integral min(a*t(z), b*t(z-d)) dz, vectorized over a and b."""
    d = float(separation)
    if not 0 <= d <= 2:
        raise ValueError('Require 0 <= separation <= 2')
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    if np.any(a < 0) or np.any(b < 0):
        raise ValueError('Density weights must be nonnegative')
    result = np.zeros_like(a)
    knots = sorted({d-1., 1., *[v for v in [0., d] if d-1. < v < 1.]})
    for left, right in zip(knots[:-1], knots[1:]):
        f0, f1 = a * max(1-abs(left), 0), a * max(1-abs(right), 0)
        g0, g1 = b * max(1-abs(left-d), 0), b * max(1-abs(right-d), 0)
        delta0, delta1 = f0-g0, f1-g1
        crossing = delta0 * delta1 < 0
        correction = np.divide(np.abs(delta0*delta1), np.abs(delta0)+np.abs(delta1),
                               out=np.zeros_like(a), where=crossing)
        result += (right-left)/2 * (np.minimum(f0, g0)+np.minimum(f1, g1)+correction)
    return result


@lru_cache(maxsize=8)
def _legendre_rule(order):
    return roots_legendre(order)


@lru_cache(maxsize=1024)
def three_feature_overlap(separation, order=256):
    """Tensor Gauss integration in x/y after exact integration in z.

    Split at triangular knots. Order is per smooth x/y segment, not total.
    Calibrations are checked at doubled order and by independent 3D integration.
    """
    d = float(separation)
    if not 0 <= d <= 2:
        raise ValueError('Require 0 <= separation <= 2')
    if d == 0:
        return 1.
    if d == 2:
        return 0.
    roots, weights = _legendre_rule(order)
    knots = sorted({d-1., 1., *[v for v in [0., d] if d-1. < v < 1.]})
    nodes, quadrature_weights = [], []
    for left, right in zip(knots[:-1], knots[1:]):
        nodes.append((left+right)/2 + (right-left)/2*roots)
        quadrature_weights.append((right-left)/2*weights)
    x, w = np.concatenate(nodes), np.concatenate(quadrature_weights)
    f, g = np.maximum(1-np.abs(x), 0), np.maximum(1-np.abs(x-d), 0)
    integrals = weighted_triangular_overlap(f[:,None]*f[None,:], g[:,None]*g[None,:], d)
    return float(np.sum(w[:,None]*w[None,:]*integrals))
