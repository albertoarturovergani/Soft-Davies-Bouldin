"""Published Soft Davies-Bouldin index and a crisp Davies-Bouldin reference."""

import numpy as np


def _matrix(value, name):
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite two-dimensional array")
    return array


def soft_davies_bouldin(X, centers, membership, *, q=1.0, membership_power=1.0, separation_tol=1e-12):
    """Evaluate SDB; rows of X and membership are samples, rows of centers are clusters.

    membership_power=1 reproduces equations (5)-(8) in Vergani & Binaghi (2018).
    membership_power=m optionally reproduces the membership-power choice in
    a historical MATLAB implementation, which differs from published eq. (7).
    Smaller values indicate better separation under this index.
    """
    X = _matrix(X, "X")
    centers = _matrix(centers, "centers")
    membership = _matrix(membership, "membership")
    n, d = X.shape
    k = centers.shape[0]
    if n < 2 or d < 1 or k < 2 or centers.shape[1] != d or membership.shape != (n, k):
        raise ValueError("Expected X (n,d), centers (k,d), membership (n,k), n>=2 and k>=2")
    if not np.isfinite(q) or q < 1 or not np.isfinite(membership_power) or membership_power <= 0:
        raise ValueError("q must be >=1 and membership_power must be >0")
    if not np.isfinite(separation_tol) or separation_tol < 0:
        raise ValueError("separation_tol must be nonnegative and finite")
    if np.any(membership < 0) or np.any(membership > 1):
        raise ValueError("Memberships must lie in [0,1]")
    if not np.allclose(np.sum(membership, axis=1), 1.0, atol=1e-7, rtol=0):
        raise ValueError("Each sample's memberships must sum to one")
    radius = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
    scatter = np.mean(radius ** q, axis=0) ** (1.0 / q)
    mean_membership = np.mean(membership ** membership_power, axis=0)
    weighted_scatter = scatter * mean_membership
    between = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    off_diagonal = ~np.eye(k, dtype=bool)
    if np.any(between[off_diagonal] <= separation_tol):
        raise ValueError("Some cluster centers coincide or are too close")
    between[~off_diagonal] = 1.0
    similarity = (weighted_scatter[:, None] + weighted_scatter[None, :]) / between
    similarity[~off_diagonal] = -np.inf
    return float(np.mean(np.max(similarity, axis=1)))


def davies_bouldin_crisp(X, labels, *, separation_tol=1e-12):
    """Classic DB with average Euclidean within-cluster distances (q=1)."""
    X = _matrix(X, "X")
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.shape[0] != X.shape[0]:
        raise ValueError("labels must be a vector of length n_samples")
    classes, inverse = np.unique(labels, return_inverse=True)
    k = classes.size
    if k < 2:
        raise ValueError("Classic DB requires at least two nonempty clusters")
    centers = np.vstack([np.mean(X[inverse == i], axis=0) for i in range(k)])
    within = np.array([np.mean(np.linalg.norm(X[inverse == i] - centers[i], axis=1)) for i in range(k)])
    between = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    off_diagonal = ~np.eye(k, dtype=bool)
    if np.any(between[off_diagonal] <= separation_tol):
        raise ValueError("Some crisp cluster centroids coincide or are too close")
    between[~off_diagonal] = 1.0
    similarity = (within[:, None] + within[None, :]) / between
    similarity[~off_diagonal] = -np.inf
    return float(np.mean(np.max(similarity, axis=1)))
