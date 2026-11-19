"""Exact counterexamples and invariances that explain the diagnostic findings."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/python'))
from sdb import soft_davies_bouldin
from diagnose_sdb_undercounting import noiseless_cases


@pytest.mark.parametrize('classes,labels,true_score,merged_score', [
    (4, [0,1,1,2], 5/8, 7/12),
    (5, [0,1,1,1,2], 81/125, 14/25),
])
def test_exact_noiseless_counterexamples(classes, labels, true_score, merged_score):
    x = np.arange(classes, dtype=float)[:,None]
    labels = np.asarray(labels)
    centers = np.array([x[labels == i].mean(axis=0) for i in range(3)])
    assert soft_davies_bouldin(x, x, np.eye(classes)) == pytest.approx(true_score)
    assert soft_davies_bouldin(x, centers, np.eye(3)[labels]) == pytest.approx(merged_score)
    assert merged_score < true_score
    # Replication in 100 coordinates and arbitrarily wider gaps do not remove
    # the noiseless counterexample: numerator and denominator scale equally.
    assert soft_davies_bouldin(np.repeat(17*x+31,100,axis=1),
                              np.repeat(17*centers+31,100,axis=1),
                              np.eye(3)[labels]) == pytest.approx(merged_score)


def test_same_membership_column_means_hide_point_assignment_changes():
    x = np.arange(4, dtype=float)[:,None]
    correct = np.eye(4)
    incorrect = correct[[1,0,3,2]]
    assert soft_davies_bouldin(x,x,correct) == soft_davies_bouldin(x,x,incorrect)


def test_exhaustive_geometry_control_and_partition_counts():
    frame = noiseless_cases()
    # Bell numbers: 15 and 52, excluding the one-cluster partition.
    for classes, number in [(4,14),(5,51)]:
        for geometry in ['line','regular_simplex']:
            sub = frame[(frame.classes == classes)&(frame.geometry == geometry)]
            assert len(sub) == number and sub.labels.nunique() == number
        simplex = frame[(frame.classes == classes)&(frame.geometry == 'regular_simplex')]
        assert simplex.loc[simplex.SDB.idxmin(),'k'] == classes
        line = frame[(frame.classes == classes)&(frame.geometry == 'line')]
        assert line.loc[line.SDB.idxmin(),'k'] == (2 if classes == 4 else 3)
