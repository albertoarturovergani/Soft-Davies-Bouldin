# 4-class overlap sweep — supplementary analysis

Entirely synthetic one-variable data: 4 equal-prior triangular classes, 11 adjacent-pair overlap levels from 100% to 0%, 20 dataset replicates, 100 starts per K=2–6. Each class has 100 training and 2,000 independent test observations. There are 110,000 configurations and 330,000 finite index values.

At disjointness, DB/k-means, DB/hardened FCM and SDB/FCM select K=4 in **20/20**, **20/20** and **0/20** datasets respectively. See [Technical note 3](../../docs/TECHNICAL_NOTES_FOUR_FIVE_CLASSES.md) for the full sweep and interpretation; all new figures use linear axes.

`per_initialization.csv.gz` contains all 110,000 synthetic configurations. `mean_scores.csv`, `selected_k.csv` and `sweep_summary.csv` contain start means, per-dataset K choices and sweep frequencies. `best_objective_partitions.csv` and `k4_performance.csv` contain independent-test metrics. `example_datasets.csv` contains the first replicate and true posteriors. `separation_sensitivity.csv` contains fixed-K score/rank correlations. `diagnostics.json` and `provenance.json` record fitting checks, settings, versions and hashes.

The `convergence_sensitivity*` files extend 87 capped FCM runs to 10,000 iterations; 2/660 K choices change. K=1 is undefined: selection frequencies at complete overlap are not correct-count recovery. No original fMRI or participant data appear here.
