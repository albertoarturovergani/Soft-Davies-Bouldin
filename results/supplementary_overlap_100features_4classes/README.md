# One hundred features, 4 classes — supplementary analysis

One hundred informative triangular coordinates, independent noise within class, equal priors and collinear centers. Eleven calibrated joint-overlap levels (100%, 90%, …, 0%), 20 independent datasets and 100 starts per K=2–6 give **110,000 configurations**. Each class supplies 100 training and 2,000 held-out samples. All new plot axes are linear.

At disjointness, the mean-index rules recover K=4 in **20/20 for DB/k-means**, **20/20 for DB/hardened FCM** and **0/20 for SDB/FCM**. See [Technical note 7](../../docs/TECHNICAL_NOTES_HUNDRED_FEATURES.md) for the full sweep and comparison with earlier feature counts.

The original FCM/SDB implementation is unchanged. There are 104,000 nearly coincident FCM fits, and undefined-score counts are {'DB_kmeans': 0, 'DB_fcm_hard': 0, 'SDB_fcm': 31}. Undefined scores remain +inf and are included in every 100-start mean; selected_k=0 means no eligible finite-mean candidate. Distinct-center screening is a separate diagnostic, not the primary rule.

No FCM fit reached the 1,000-iteration cap, so no 10,000-iteration refit was needed. The convergence files record this zero-refit audit; they are not an independent longer-iteration test.

The bundle contains the full synthetic score grid, mean-index choices, held-out performance, examples with all 100 coordinates, overlap calibration and independent numerical checks, degeneracy diagnostics, convergence sensitivity and source/data hashes. No participant or fMRI data are included.
