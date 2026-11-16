# Mathematical definition and implementation decisions

This repository implements equations (5)–(8) in Vergani & Binaghi, *A Soft Davies-Bouldin Separation Measure*, FUZZ-IEEE 2018.

Let `X ∈ R^(N×d)`, fuzzy centers `A ∈ R^(C×d)`, memberships `U ∈ [0,1]^(N×C)` with `∑_i U[x,i]=1`, and `C≥2`.

For a chosen scatter exponent `q≥1` (default 1) and Euclidean inter-center separation:

```text
S[i] = (mean_x ||X[x] - A[i]||_2^q)^(1/q)
u_bar[i] = mean_x U[x,i]
D[i,j] = ||A[i] - A[j]||_2
R[i,j] = (S[i] * u_bar[i] + S[j] * u_bar[j]) / D[i,j]
SDB = mean_i max_{j != i} R[i,j]
```

Unlike crisp DB, the scatter `S[i]` averages distances over **all N samples**, not only samples assigned to cluster `i`. This matters greatly when interpreting the scores. `U` contributes through average cluster memberships as published. Centroids and memberships may come from any fuzzy partition algorithm, not necessarily FCM.

The optional `membership_power=m` replaces `mean(U)` with `mean(U**m)`, matching a historical MATLAB function found in the author archive. It is useful for sensitivity analysis, but it is *not Eq. (7)* and it is *not* the default. The FCM fuzzifier `m` is independent of the index's membership power parameter.

The API rejects zero-separation cluster-center pairs. It also rejects C=1, non-finite inputs, malformed membership matrices, and probabilities outside [0,1]. The default uses Euclidean distance for both scatter and center separation.

A low SDB is a low cluster-similarity score **within this definition**. Direct magnitude comparisons with the crisp DB index are not accuracy claims; examine known labels and downstream clustering criteria. The 2018 empirical protocol includes three distinct summaries: AIM (minimize across candidate C), RIM (score at ground-truth C), and MRC (median trend across datasets).

The supplied FCM implementation optimizes the standard squared-distance fuzzy c-means objective. Initialization and convergence are independent of the historical MATLAB toolbox and can change individual random runs even when the index implementation is identical.
