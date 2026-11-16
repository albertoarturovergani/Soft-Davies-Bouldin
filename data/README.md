# Datasets and provenance

The 2018 FUZZ-IEEE paper evaluated **eight known-ground-truth benchmarks** and four derivative fMRI representations. This repository initially supports the eight benchmarks. The original binary datasets below were recovered from the author's 2018 research archives, and their shapes/labels were independently inspected on 2026-11-16.

| Benchmark | Original file | Samples | Features | Classes |
|---|---|---:|---:|---:|
| Cancer | cancer.mat | 699 | 9 | 2 |
| Crab | crab.mat | 200 | 6 | 2 |
| Glass | glass.mat | 214 | 9 | 2 |
| Iris | iris.mat | 150 | 4 | 3 |
| Ovarian | ovarian.mat | 216 | 100 | 2 |
| Simple | simpleclusters.mat | 1000 | 2 | 4 |
| Thyroid | thyroid.mat | 7200 | 21 | 3 |
| Wine | wine.mat | 178 | 13 | 3 |

**The archived Glass version uses two ground-truth categories**; do not substitute the multiclass form of a modern public Glass dataset without explaining the label transformation. The archived Cancer values are already transformed, so using a contemporary public repository download may also differ from the original analysis.

The expected MAT layout is `Data` (N×D observations) and `T` (C×N one-hot labels). `scripts/python/benchmark.py` reads these files, and `scripts/matlab/benchmark_original.m` reads the same files directly. Optionally, `scripts/python/convert_original_mat.py` creates CSVs locally for inspection. Place the data in `data/original/` *on your machine only*.

The repository intentionally does not commit the original `.mat` files. Data hosting licenses and original acquisition provenance must be audited individually before redistributing original or derived datasets publicly. UCI now uses CC BY 4.0 for newly donated datasets, but that policy alone does not establish redistribution rights for every legacy or separately sourced dataset. For an immediately runnable public demo, use the scikit-learn Iris/Wine fallback or the small synthetic fixture.

The original paper PDF is also linked via DOI rather than embedded in the public repository because the author's IEEE download states reuse restrictions. A private reference copy is kept separately in the user's Drive project `docs` folder.
