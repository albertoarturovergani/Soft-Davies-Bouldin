# Datasets and provenance

The 2018 FUZZ-IEEE paper evaluated **eight known-ground-truth benchmarks** and four derivative fMRI representations. This repository supports independent reanalysis of the eight benchmarks and, when user-supplied source files are available, the REST/TASK fMRI inputs. The original binary datasets below were recovered from the author's 2018 research archives, and their shapes/labels were independently inspected on 2026-11-16.

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

To reproduce the fMRI experiment, additionally supply `dataset_rest_rio.mat` (19 subjects with 96×175 ROI-time arrays) and `dataset_task_208_rio.mat` (15 subjects with 96×208 arrays). The `scripts/python/reproduce_abstract.py` script processes both ROI/TOI orientations. The historical 10-start result directory contains pseudonymized index scores. The new 100-start extension publishes only aggregates and figures; it does not add individual scores or source fMRI time series.

The repository intentionally does not commit the original `.mat` files. Data hosting licenses and original acquisition provenance must be audited individually before redistributing original or derived datasets publicly. UCI now uses CC BY 4.0 for newly donated datasets, but that policy alone does not establish redistribution rights for every legacy or separately sourced dataset. For an immediately runnable public demo, use the scikit-learn Iris/Wine fallback or the small synthetic fixture.

The [100-initialization result bundle](../results/reproduction_100seeds/) contains aggregates derived from 7,200 benchmark and 61,200 fMRI configurations, plus provenance. All new per-subject/per-initialization score files remain local and are excluded from Git. The 10-versus-100 comparison uses the same local inputs and environment; see the [updated report](../docs/FMRI_100_INITIALIZATIONS.md).

The original paper PDF is also linked via DOI rather than embedded in the public repository because the author's IEEE download states reuse restrictions. A private reference copy is kept separately in the user's Drive project `docs` folder.
