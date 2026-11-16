"""Convert private/local 2018 Data/T MAT files to cross-language CSV files."""

import argparse
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from benchmark import DATASETS, ORIGINAL_FILES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/original"))
    parser.add_argument("--output", type=Path, default=Path("data/benchmarks"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name in DATASETS:
        source = args.source / ORIGINAL_FILES[name]
        if not source.exists():
            print(f"Skipping {name}: missing {source}")
            continue
        record = loadmat(source, variable_names=["Data", "T"])
        X, T = np.asarray(record["Data"], dtype=float), np.asarray(record["T"])
        if T.ndim != 2 or X.ndim != 2 or T.shape[1] != X.shape[0] or not np.all(T.sum(axis=0) == 1):
            raise ValueError(f"Bad Data/T layout in {source}")
        out = args.output / f"{name}.csv"
        np.savetxt(out, np.column_stack([X, np.argmax(T, axis=0)]), delimiter=",", fmt="%.16g")
        print(f"{name}: X={X.shape}, classes={T.shape[0]} -> {out}")
    print("Converted files remain untracked. Check third-party licenses before redistributing them.")


if __name__ == "__main__":
    main()
