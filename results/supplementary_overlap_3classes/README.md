# Three-class overlap sweep — supplementary analysis

Entirely synthetic, one-variable data with three equally likely triangular classes. Eleven adjacent-pair overlap levels (100%, 90%, ..., 0%), 20 independent datasets, 100 observations per class, 100 initializations for each K=2–6: 110,000 configurations and 330,000 index values. FCM m=2 is fixed.

See [Technical note 2](../../docs/TECHNICAL_NOTES_THREE_CLASSES.md) for definitions, interpretation, convergence sensitivity and reproduction commands. All new figures use linear axes. At 20% overlap SDB recovers K=3 in 13/20 datasets versus 5/20 for both DB criteria; with disjoint classes SDB recovers it in 15/20 versus 20/20 for both DB criteria. SDB does not retain an advantage across the full sweep.

- `per_initialization.csv.gz`: all synthetic scores and fitting diagnostics, 110,000 rows.
- `mean_scores.csv`, `selected_k.csv`, `sweep_summary.csv`: means over 100 starts, per-dataset K selections, and summary frequencies.
- `best_objective_partitions.csv`, `k3_performance.csv`: independent-test evaluation of the best native training-objective partition for each algorithm/K.
- `example_datasets.csv`: replicate 0 observations and true posteriors at all eleven levels.
- `separation_sensitivity.csv`: within-replicate fixed-K score/rank correlations.
- `diagnostics.json`, `provenance.json`: completeness, convergence, configuration, versions and hashes.
- `convergence_sensitivity*`: 123 capped FCM fits extended to 10,000 iterations; no remaining caps and no changed K choices.

At complete overlap the observable count is one, which DB/SDB cannot evaluate. Saved K=3 frequencies at that endpoint are not correct-count recovery. Posterior MSE and Brier components are averaged over observations and class columns. This bundle contains no original fMRI or participant data.
