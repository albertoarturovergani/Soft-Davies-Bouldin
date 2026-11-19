"""Joint overlap of triangular product densities via likelihood-ratio convolution.

On the common support, Y=log[t(X-d)/t(X)] is strictly increasing in X.
Its conditional CDF has an analytic inverse. Independent coordinates add Y;
reflection symmetry gives overlap = 2 * M**D * P(sum(Y) > 0), where M is
the one-coordinate common-support probability under the first density.
FFT bin convolution uses half the central-bin mass at the zero boundary.
Grid refinement, domain widening and independent conditional Sobol quadrature
check this numerical approximation; none is claimed as a rigorous error bound.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.fft import fftshift, ifftshift, irfft, rfft
from scipy.optimize import brentq
from scipy.stats import qmc

from ten_feature_geometry import conditional_overlap

FEATURES = 100
POWER = 19
EXTENT = 64.
REFERENCE_POWER = 21
REFERENCE_SEEDS = [519191, 519192, 519193, 519194]
QMC_POWER = 18


def common_mass(d):
    return 1-d*d/2 if d < 1 else (2-d)**2/2


def log_ratio_cdf(y, separation):
    """CDF conditional on X in [d-1,1], with X drawn from t(x)."""
    d = float(separation)
    if not 0 < d < 2:
        raise ValueError('Require 0 < separation < 2')
    r = np.exp(np.asarray(y, float))
    x = (r+d-1)/(1+r)
    if d < 1:
        left, right = r < 1-d, r > 1/(1-d)
        x = np.where(left, d/np.where(left,1-r,1)-1, x)
        x = np.where(right, 1-d/np.where(right,r-1,1), x)
        f = np.where(x < 0, (x+1)**2/2, 1-(1-x)**2/2)
        answer = (f-d*d/2)/common_mass(d)
    else:
        answer = 1-((1-x)/(2-d))**2
    return np.clip(answer,0,1)


def product_overlap(separation, features=FEATURES, *, power=POWER, extent=EXTENT):
    """Numerical integral min(f,g), using a discretized log-ratio sum."""
    d = float(separation)
    if not np.isfinite(d) or not 0 <= d <= 2:
        raise ValueError('Require 0 <= separation <= 2')
    if not isinstance(features,int) or features < 1 or power < 10 or extent <= 0:
        raise ValueError('Invalid dimension or convolution grid')
    if d == 0:
        return 1.
    if d == 2:
        return 0.
    n = 2**power
    edges = (np.arange(n+1)-n//2-.5)*(2*extent/n)
    mass = np.diff(log_ratio_cdf(edges,d))
    if mass.min() < -1e-12 or abs(mass.sum()-1) > 1e-10:
        raise ArithmeticError('Invalid or insufficiently wide log-ratio grid')
    mass = np.maximum(mass,0)
    total = fftshift(irfft(rfft(ifftshift(mass))**features,n=n))
    value = float(2*common_mass(d)**features*(total[n//2+1:].sum()+total[n//2]/2))
    if not -1e-10 <= value <= 1+1e-10:
        raise ArithmeticError('Overlap outside probability range')
    return float(np.clip(value,0,1))


def conditional_reference(shifts, *, features=FEATURES, power=QMC_POWER, seed=519191, block_power=14):
    """Independent D-1 dimensional quadrature, bounded-memory Sobol blocks."""
    if power < block_power:
        block_power = power
    sampler = qmc.Sobol(d=features-1,scramble=True,seed=seed)
    result = np.zeros(len(shifts))
    blocks = 2**(power-block_power)
    for _ in range(blocks):
        unit = sampler.random(2**block_power)
        result += np.array([conditional_overlap(d,unit) for d in shifts])/blocks
    return result


def calibrate(directory):
    directory = Path(directory)
    csv, meta = directory/'overlap_calibration.csv', directory/'calibration.json'
    settings = dict(features=FEATURES,fft_power=POWER,fft_extent=EXTENT,
                    reference_fft_power=REFERENCE_POWER,reference_extent=128.,
                    qmc_power=QMC_POWER,qmc_seeds=REFERENCE_SEEDS,
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if meta.exists() and csv.exists():
        saved = json.loads(meta.read_text())
        if saved['settings'] != settings:
            raise ValueError('Calibration settings/source mismatch')
        frame = pd.read_csv(csv)
        frame['per_feature_shift'] = saved['exact_shifts']
        return frame
    target = 1-np.arange(11)/10
    shifts = [0.] + [brentq(lambda d:product_overlap(d)-o,0,2,xtol=1e-10)
                     for o in target[1:-1]] + [2.]
    reference = np.array([product_overlap(d,power=REFERENCE_POWER) for d in shifts])
    wide = np.array([product_overlap(d,power=REFERENCE_POWER+1,extent=128.) for d in shifts])
    if np.max(np.abs(reference-target)) >= 2e-5 or np.max(np.abs(wide-reference)) >= 1e-8:
        raise ArithmeticError('FFT calibration refinement/domain check failed')
    print('FFT calibration passed; checking independent 99-dimensional quadrature',flush=True)
    estimates = []
    for seed in REFERENCE_SEEDS:
        estimates.append(conditional_reference(shifts,seed=seed))
        print(f'Independent calibration scramble {len(estimates)}/4 complete',flush=True)
    values = np.asarray(estimates)
    mean, se = values.mean(axis=0),values.std(axis=0,ddof=1)/2
    if np.max(np.abs(mean-reference)) >= .0015:
        raise ArithmeticError('Independent conditional quadrature disagrees with convolution')
    frame = pd.DataFrame(dict(step=np.arange(11),target_joint_overlap=target,
                             per_feature_shift=shifts,euclidean_center_distance=np.sqrt(FEATURES)*np.array(shifts),
                             per_feature_overlap=(1-np.array(shifts)/2)**2,
                             reference_joint_overlap=reference,reference_absolute_error=np.abs(reference-target),
                             widened_domain_overlap=wide,qmc_reference_overlap=mean,qmc_scramble_se=se,
                             qmc_reference_absolute_error=np.abs(mean-reference)))
    metadata = dict(settings=settings,exact_shifts=shifts,fft_tolerance=2e-5,domain_tolerance=1e-8,
                    qmc_check_tolerance=.0015,qmc_estimates=values.tolist(),root_tolerance=1e-10,
                    maximum_fft_refinement_error=float(np.max(np.abs(reference-target))),
                    maximum_domain_difference=float(np.max(np.abs(wide-reference))),
                    maximum_qmc_difference=float(np.max(np.abs(mean-reference))),
                    method=__doc__)
    directory.mkdir(parents=True,exist_ok=True)
    frame.to_csv(csv,index=False,float_format='%.17g')
    meta.write_text(json.dumps(metadata,indent=2)+'\n')
    return frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args = parser.parse_args()
    print(calibrate(args.output_dir).to_string(index=False),flush=True)
