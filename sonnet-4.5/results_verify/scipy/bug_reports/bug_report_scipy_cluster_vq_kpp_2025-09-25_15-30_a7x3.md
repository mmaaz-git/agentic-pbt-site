# Bug Report: scipy.cluster.vq k-means++ Division by Zero

**Target**: `scipy.cluster.vq._kpp`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The k-means++ initialization method (`_kpp` function) performs division by zero when selecting centroids from data containing duplicate points, producing a `RuntimeWarning: invalid value encountered in divide`.

## Property-Based Test

```python
import numpy as np
import pytest
from hypothesis import given, strategies as st, settings
from scipy.cluster.vq import kmeans2


@given(st.integers(min_value=2, max_value=10),
       st.integers(min_value=1, max_value=5))
@settings(max_examples=100)
def test_kmeans2_kpp_handles_duplicate_data(n_duplicates, n_features):
    data = np.tile([1.0] * n_features, (n_duplicates, 1))

    centroid, label = kmeans2(data, min(2, n_duplicates), minit='++')

    assert not np.any(np.isnan(centroid))
```

**Failing input**: Any data with all identical points, e.g., `data = [[1, 2], [1, 2], [1, 2], [1, 2]]` with `k=2`

## Reproducing the Bug

```python
import numpy as np
import warnings
from scipy.cluster.vq import kmeans2

data = np.array([
    [1.0, 2.0],
    [1.0, 2.0],
    [1.0, 2.0],
    [1.0, 2.0],
])

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    centroid, label = kmeans2(data, 2, minit='++')

    for warning in w:
        if "invalid value encountered in divide" in str(warning.message):
            print(f"BUG: Division by zero at {warning.filename}:{warning.lineno}")
```

## Why This Is A Bug

The k-means++ algorithm (Arthur & Vassilvitskii, 2007) selects each centroid with probability proportional to its squared distance from the nearest already-selected centroid. When all remaining data points are identical to already-selected centroids, all squared distances are 0, making the sum of distances also 0.

In `scipy/cluster/vq.py:616`, the code computes:
```python
D2 = cdist(init[:i,:], data, metric='sqeuclidean').min(axis=0)
probs = D2/D2.sum()  # Division by zero when D2.sum() == 0
```

This causes a `RuntimeWarning` and produces NaN values in `probs`, which could lead to undefined behavior when selecting the next centroid.

## Fix

```diff
--- a/scipy/cluster/vq.py
+++ b/scipy/cluster/vq.py
@@ -613,7 +613,13 @@ def _kpp(data, k, rng, xp):
             data_idx = rng_integers(rng, data.shape[0])
         else:
             D2 = cdist(init[:i,:], data, metric='sqeuclidean').min(axis=0)
-            probs = D2/D2.sum()
+            d2_sum = D2.sum()
+            if d2_sum == 0:
+                # All points are at distance 0 from selected centroids
+                # Select uniformly at random
+                probs = np.ones(len(D2)) / len(D2)
+            else:
+                probs = D2/d2_sum
             cumprobs = probs.cumsum()
             r = rng.uniform()
             cumprobs = np.asarray(cumprobs)
```