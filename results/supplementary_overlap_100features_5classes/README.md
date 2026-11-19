# One hundred features, 5 classes — supplementary analysis

One hundred informative triangular coordinates, independent noise within class, equal priors and collinear centers. Eleven calibrated joint-overlap levels (100%, 90%, …, 0%), 20 independent datasets and 100 starts per K=2–6 give **110,000 configurations**. Each class supplies 100 training and 2,000 held-out samples. All new plot axes are linear.

At disjointness, the mean-index rules recover K=5 in **19/20 for DB/k-means**, **13/20 for DB/hardened FCM** and **0/20 for SDB/FCM**. See [Technical note 7](../../docs/TECHNICAL_NOTES_HUNDRED_FEATURES.md) for the full sweep and comparison with earlier feature counts.

The original FCM/SDB implementation is unchanged. There are 102,064 nearly coincident FCM fits, and undefined-score counts are {'DB_kmeans': 0, 'DB_fcm_hard': 0, 'SDB_fcm': 72}. Undefined scores remain +inf and are included in every 100-start mean; selected_k=0 means no eligible finite-mean candidate. Distinct-center screening is a separate diagnostic, not the primary rule.

Extending 43 capped FCM fits to 10,000 iterations leaves 0 capped and changes 0/660 K choices, including 0/220 SDB choices. Refits reconstruct the exact shifts from provenance JSON. Primary files retain the 1,000-iteration protocol.

The bundle contains the full synthetic score grid, mean-index choices, held-out performance, examples with all 100 coordinates, overlap calibration and independent numerical checks, degeneracy diagnostics, convergence sensitivity and source/data hashes. No participant or fMRI data are included.
