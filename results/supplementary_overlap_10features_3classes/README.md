# Ten features, 3 classes — supplementary analysis

Synthetic triangular product densities with ten informative coordinates, independent noise within class, equal priors and collinear centers. Eleven nominal joint-overlap levels (100%, 90%, …, 0%), 20 independent datasets and 100 initializations per K=2–6 give **110,000 configurations**. Each class has 100 training and 2,000 independent test samples. All new plots have linear axes.

At disjointness, the mean-index rules recover K=3 in **20/20 for DB/k-means**, **20/20 for DB/hardened FCM**, and **20/20 for SDB/FCM**. See [Technical note 6](../../docs/TECHNICAL_NOTES_TEN_FEATURES.md) for the full sweep, comparisons, validity checks and limitations.

The original FCM/SDB implementation is unchanged. Near-coincident centers occur in 92,664 fits. Undefined score counts are {'DB_kmeans': 0, 'DB_fcm_hard': 0, 'SDB_fcm': 6885}. They are retained as +inf with explicit reasons; no failed initialization is silently discarded from the mean. `selected_k=0` means no eligible finite-mean candidate. `distinct_clusters_selected_k.csv` is a separate conservative validity sensitivity, not the primary result.

Extending 3,964 capped FCM fits to 10,000 iterations leaves 61 at the cap and changes 0/660 selections, including 0/220 SDB selections. Primary data retain the 1,000-iteration protocol.

Files include full synthetic scores, native-objective best partitions, mean-index choices, fixed-K held-out performance, all ten coordinates of example datasets, independent overlap-calibration checks, degeneracy diagnostics, convergence refits and source/data hashes. No participant or fMRI data are included.
