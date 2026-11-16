"""Small NumPy implementation of fuzzy c-means for reproducible examples."""

import numpy as np
from .index import _matrix


def fuzzy_c_means(X, k, *, m=2.0, seed=42, max_iter=300, tol=1e-6):
    """Return (centers[k,d], membership[n,k], iterations) for fuzzy c-means."""
    X = _matrix(X, "X")
    n, d = X.shape
    if n < 2 or d < 1 or not isinstance(k, (int, np.integer)) or not 2 <= k <= n:
        raise ValueError("k must be an integer in [2,n_samples]")
    if not np.isfinite(m) or m <= 1 or not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("m must be >1 and max_iter must be a positive integer")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be positive and finite")
    rng = np.random.default_rng(seed)
    U = rng.uniform(size=(n, k))
    U /= U.sum(axis=1, keepdims=True)
    centers = np.zeros((k, d), dtype=float)
    for iteration in range(1, max_iter + 1):
        powers = U ** m
        centers = (powers.T @ X) / powers.sum(axis=0)[:, None]
        distances = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        new_U = np.zeros_like(U)
        zero = distances == 0
        rows_with_zero = zero.any(axis=1)
        if np.any(rows_with_zero):
            new_U[rows_with_zero] = zero[rows_with_zero] / zero[rows_with_zero].sum(axis=1, keepdims=True)
        if np.any(~rows_with_zero):
            log_weights = -2.0 / (m - 1.0) * np.log(distances[~rows_with_zero])
            log_weights -= log_weights.max(axis=1, keepdims=True)
            weights = np.exp(log_weights)
            new_U[~rows_with_zero] = weights / weights.sum(axis=1, keepdims=True)
        delta = np.max(np.abs(new_U - U))
        U = new_U
        if delta < tol:
            break
    powers = U ** m
    centers = (powers.T @ X) / powers.sum(axis=0)[:, None]
    return centers, U, iteration
