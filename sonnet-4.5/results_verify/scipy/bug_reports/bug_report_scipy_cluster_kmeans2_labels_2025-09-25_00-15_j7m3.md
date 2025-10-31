# Bug Report: scipy.cluster.vq.kmeans2 Returns Inconsistent Labels and Centroids

**Target**: `scipy.cluster.vq.kmeans2`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `kmeans2` function returns labels that do not correspond to the returned centroids. The labels represent assignments to the centroids from the second-to-last iteration, not the final centroids.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from scipy.cluster.vq import kmeans2

@given(st.integers(min_value=10, max_value=30),
       st.integers(min_value=2, max_value=5),
       st.integers(min_value=2, max_value=5))
@settings(max_examples=50, deadline=10000)
def test_kmeans2_labels_match_codebook(n_obs, n_features, k):
    assume(n_obs >= k)
    rng = np.random.default_rng(42)
    data = rng.standard_normal((n_obs, n_features))

    centroid, label = kmeans2(data, k, iter=5, minit='points',
                              rng=np.random.default_rng(123))

    for obs_idx in range(n_obs):
        assigned_label = label[obs_idx]
        assigned_centroid = centroid[assigned_label]

        distances = np.linalg.norm(data[obs_idx] - centroid, axis=1)
        closest_idx = np.argmin(distances)

        assert assigned_label == closest_idx, \
            f"Obs {obs_idx} assigned to {assigned_label}, but closest is {closest_idx}"
```

**Failing input**: `n_obs=27, n_features=2, k=2`

## Reproducing the Bug

```python
import numpy as np
from scipy.cluster.vq import kmeans2

rng = np.random.default_rng(42)
data = rng.standard_normal((27, 2))

centroid, label = kmeans2(data, 2, iter=5, minit='points',
                          rng=np.random.default_rng(123))

obs_idx = 7
assigned_label = label[obs_idx]
distances = np.linalg.norm(data[obs_idx] - centroid, axis=1)
closest_idx = np.argmin(distances)

print(f"Observation {obs_idx}: {data[obs_idx]}")
print(f"Assigned label: {assigned_label}")
print(f"Closest centroid: {closest_idx}")
print(f"Distances: {distances}")
```

Output shows the observation is assigned to centroid 0, but is actually closer to centroid 1.

## Why This Is A Bug

The fundamental guarantee of k-means is that each observation is assigned to its nearest centroid. The returned labels should correspond to the returned centroids, but they don't.

Looking at the source code (vq.py lines 821-832):

```python
for _ in range(iter):
    # Compute labels for current code_book
    label = vq(data, code_book, check_finite=check_finite)[0]
    # Update code_book based on labels
    new_code_book, has_members = _vq.update_cluster_means(data, label, nc)
    ...
    code_book = new_code_book

return xp.asarray(code_book), xp.asarray(label)
```

The problem: after the last iteration, `code_book` is updated (line 830) but `label` is not recomputed. The returned labels correspond to the code_book from iteration N-1, while the returned code_book is from iteration N.

## Fix

Recompute labels after the loop to ensure they match the final centroids:

```diff
--- a/scipy/cluster/vq.py
+++ b/scipy/cluster/vq.py
@@ -828,7 +828,10 @@ def kmeans2(data, k, iter=10, thresh=1e-5, minit='random',
             # Set the empty clusters to their previous positions
             new_code_book[~has_members] = code_book[~has_members]
         code_book = new_code_book
-
+
+    # Recompute labels to match the final code_book
+    label = vq(data, code_book, check_finite=check_finite)[0]
+
     return xp.asarray(code_book), xp.asarray(label)
```