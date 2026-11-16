from sklearn.datasets import load_iris
from sdb import soft_davies_bouldin, davies_bouldin_crisp, fuzzy_c_means
import numpy as np

X = load_iris().data
centers, membership, iterations = fuzzy_c_means(X, 3, m=2, seed=42)
labels = np.argmax(membership, axis=1)
print(f"FCM iterations: {iterations}")
print(f"SDB, published Eq. 7: {soft_davies_bouldin(X, centers, membership):.6f}")
print(f"SDB, historical power variant: {soft_davies_bouldin(X, centers, membership, membership_power=2):.6f}")
print(f"Classic DB on hard FCM labels: {davies_bouldin_crisp(X, labels):.6f}")
