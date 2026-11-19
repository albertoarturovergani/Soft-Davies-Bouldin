"""Start or resume the fixed ten-feature protocol using exact saved shifts.

The numerical solver and score policy live in supplementary_overlap_ten_features.
On resumption, configuration floats come from the checkpoint JSON, not a CSV
round trip that could change their last binary digit and invalidate the guard.
"""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform

import numpy as np

from supplementary_overlap_ten_features import run_study
from ten_feature_geometry import calibrate


def configuration_for(classes,calibration,checkpoints):
    config = dict(n_classes=classes,n_features=10,replicates=20,initializations=100,
                  steps=10,n_per_class=100,test_per_class=2000,max_k=6,max_iter=1000,tol=1e-7)
    path = checkpoints/'manifest.json'
    if path.exists():
        saved = json.loads(path.read_text())['configuration']
        if any(saved.get(key)!=value for key,value in config.items()):
            raise ValueError('Checkpoint does not match the fixed ten-feature protocol')
        if not np.allclose(saved['separations'],calibration.per_feature_shift,atol=1e-12,rtol=0):
            raise ValueError('Calibration disagrees with saved shifts')
        return saved
    config['separations'] = calibration.per_feature_shift.tolist()
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--classes',type=int,choices=[2,3,4,5],nargs='+',default=[2,3,4,5])
    parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--output-root',type=Path,default=Path('results'))
    parser.add_argument('--checkpoint-root',type=Path,required=True)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    calibration_dir = args.output_root/'supplementary_overlap_10features_calibration'
    calibration = calibrate(calibration_dir)
    # If only some classes were started previously, keep their exact calibration
    # for the not-yet-started classes too, rather than mixing CSV rounding modes.
    for classes in range(2,6):
        manifest = args.checkpoint_root/f'{classes}classes/manifest.json'
        prior = args.output_root/f'supplementary_overlap_10features_{classes}classes/provenance.json'
        if manifest.exists():
            saved = configuration_for(classes,calibration,manifest.parent)
        elif prior.exists():
            saved = json.loads(prior.read_text())['configuration']
            if saved['n_features'] != 10 or not np.allclose(saved['separations'],calibration.per_feature_shift,atol=1e-12,rtol=0):
                raise ValueError('Calibration disagrees with archived provenance')
        else:
            continue
        calibration['per_feature_shift'] = saved['separations']
        break
    environment = dict(python=platform.python_version(),
                       packages={p:version(p) for p in ['numpy','scipy','pandas','scikit-learn','matplotlib','threadpoolctl']})
    args.checkpoint_root.mkdir(parents=True,exist_ok=True)
    env_path = args.checkpoint_root/'execution_environment.json'
    if env_path.exists() and json.loads(env_path.read_text()) != environment:
        raise ValueError('Execution environment differs from the saved run')
    for classes in args.classes:
        prior = args.output_root/f'supplementary_overlap_10features_{classes}classes/provenance.json'
        if prior.exists():
            p = json.loads(prior.read_text())
            if p['python'] != environment['python'] or p['packages'] != environment['packages']:
                raise ValueError('Execution environment differs from prior study provenance')
    env_path.write_text(json.dumps(environment,indent=2)+'\n')
    for classes in args.classes:
        directory = args.output_root/f'supplementary_overlap_10features_{classes}classes'
        checkpoints = args.checkpoint_root/f'{classes}classes'
        config = configuration_for(classes,calibration,checkpoints)
        run_study(config,directory,checkpoints,args.workers)
        for name in ['overlap_calibration.csv','calibration.json']:
            (directory/name).write_bytes((calibration_dir/name).read_bytes())
        print(f'PRIMARY COMPLETE: C={classes}, D=10',flush=True)


if __name__ == '__main__':
    main()
