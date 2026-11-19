"""Calibrated joint overlap for ten independent triangular coordinates.

Integrate one coordinate exactly; scrambled Sobol quadrature covers the other
nine under t(x) restricted to the common support. No planar-boundary assumption.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import qmc

from overlap_geometry import weighted_triangular_overlap

CALIBRATION_SEED = 20261119
REFERENCE_SEEDS = [51191, 51192, 51193, 51194]


def conditional_overlap(separation, unit):
    """Importance-weighted conditional quadrature; unit has D-1 columns."""
    d = float(separation)
    if not np.isfinite(d) or not 0 <= d <= 2:
        raise ValueError('Require 0 <= separation <= 2')
    if d == 0:
        return 1.
    if d == 2:
        return 0.
    lo = d - 1
    cdflo = (lo+1)**2/2 if lo < 0 else 1-(1-lo)**2/2
    v = cdflo + (1-cdflo)*unit
    x = np.where(v < .5, np.sqrt(2*v)-1, 1-np.sqrt(2*(1-v)))
    p = np.maximum(1-np.abs(x), 0)
    q = np.maximum(1-np.abs(x-d), 0)
    ratio = np.prod(q/p, axis=1)
    return float((1-cdflo)**unit.shape[1] * weighted_triangular_overlap(1., ratio, d).mean())


def calibrate(directory):
    """Nine interior targets; independent larger scrambles check actual overlap."""
    directory = Path(directory)
    csv = directory/'overlap_calibration.csv'
    meta = directory/'calibration.json'
    if csv.exists() and meta.exists():
        metadata = json.loads(meta.read_text())
        if metadata.get('calibration_seed') != CALIBRATION_SEED or metadata.get('features') != 10:
            raise ValueError('Calibration metadata mismatch')
        frame = pd.read_csv(csv)
        assert len(frame) == 11 and frame.reference_absolute_error.max() < 2e-4
        return frame
    unit = qmc.Sobol(d=9, scramble=True, seed=CALIBRATION_SEED).random_base2(18)
    shifts = [0.] + [brentq(lambda d: conditional_overlap(d, unit)-(1-step/10),
                           0., 2., xtol=1e-9) for step in range(1, 10)] + [2.]
    estimates = []
    for seed in REFERENCE_SEEDS:
        reference = qmc.Sobol(d=9, scramble=True, seed=seed).random_base2(20)
        estimates.append([conditional_overlap(d, reference) for d in shifts])
    values = np.asarray(estimates)
    target = 1-np.arange(11)/10
    mean, se = values.mean(axis=0), values.std(axis=0, ddof=1)/2
    assert np.max(np.abs(mean-target)) < 2e-4
    frame = pd.DataFrame(dict(step=np.arange(11), target_joint_overlap=target,
                             per_feature_shift=shifts, euclidean_center_distance=np.sqrt(10)*np.array(shifts),
                             per_feature_overlap=(1-np.array(shifts)/2)**2,
                             reference_joint_overlap=mean, reference_scramble_se=se,
                             reference_absolute_error=np.abs(mean-target)))
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv, index=False, float_format='%.16g')
    metadata = dict(features=10, calibration_seed=CALIBRATION_SEED, calibration_power=18,
                    reference_seeds=REFERENCE_SEEDS, reference_power=20,
                    reference_estimates=values.tolist(), maximum_absolute_error=float(np.max(np.abs(mean-target))),
                    tolerance=2e-4, root_tolerance=1e-9,
                    method='Conditional integration: exact weighted triangular minimum in coordinate 10; nine-dimensional scrambled Sobol importance quadrature under the first density restricted to the common support. Reference SE measures variation across four independent scrambles, not a rigorous error bound.')
    meta.write_text(json.dumps(metadata, indent=2)+'\n')
    return frame
