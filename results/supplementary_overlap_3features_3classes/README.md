# Three features, 3 classes — supplementary analysis

Entirely synthetic data: 3 equal-prior products of three triangular densities. All three features are informative, with independent noise within class and centers along (1,1,1). Eleven **joint adjacent-density overlap levels** from 100% to 0%, 20 dataset replicates, 100 starts per K=2–6; 100 training and 2,000 independent test observations per class. There are 110,000 configurations and 330,000 finite index values.

At disjointness, DB/k-means, DB/hardened FCM and SDB/FCM select K=3 in **20/20**, **20/20** and **19/20** datasets. See [Technical note 5](../../docs/TECHNICAL_NOTES_THREE_FEATURES.md) for all levels and comparisons with one and two features. All axes are linear.

`per_initialization.csv.gz` contains the full synthetic score grid. `mean_scores.csv`, `selected_k.csv` and `sweep_summary.csv` contain means across starts, K choices and selection frequencies. `best_objective_partitions.csv` and `k3_performance.csv` contain independent-test evaluations. `example_datasets.csv` contains all three features and class posteriors for replicate 0 at each level. `overlap_calibration.csv` reports joint and marginal overlaps, center shifts and a doubled-order quadrature check. `separation_sensitivity.csv`, `diagnostics.json` and `provenance.json` record score trends, solver checks, settings, versions and hashes.

The `convergence_sensitivity*` files extend 1351 capped FCM runs to 10,000 iterations; 0 remain capped and 0/660 K choices change. `convergence_recovery_rates.csv` records changes in target-K frequencies (0/33). Primary results are retained. Complete-overlap target-K frequencies are descriptive only: K=1 is undefined for these criteria. No participant or fMRI data are included.
