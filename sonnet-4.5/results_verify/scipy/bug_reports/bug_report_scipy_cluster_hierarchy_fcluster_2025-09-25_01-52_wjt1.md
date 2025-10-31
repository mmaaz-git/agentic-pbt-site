# Bug Report: scipy.cluster.hierarchy.fcluster - Incorrect cluster count with identical observations

**Target**: `scipy.cluster.hierarchy.fcluster`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When using `fcluster` with the `'maxclust'` criterion on a linkage matrix where all merge distances are exactly 0 (i.e., all observations are identical), the function returns fewer clusters than requested for intermediate values of k, but returns exactly k clusters when k equals the number of observations. This inconsistent behavior violates the expected contract of the `maxclust` criterion.

## Property-Based Test

```python
from hypothesis import given, settings, assume, strategies as st
from hypothesis.extra import numpy as npst
import numpy as np
import scipy.cluster.hierarchy as hierarchy

@given(
    npst.arrays(
        dtype=np.float64,
        shape=npst.array_shapes(min_dims=2, max_dims=2, min_side=3, max_side=50),
        elements=st.floats(
            min_value=-1e6, max_value=1e6,
            allow_nan=False, allow_infinity=False
        )
    ),
    st.integers(min_value=1, max_value=10)
)
@settings(max_examples=200)
def test_fcluster_maxclust_count(obs, n_clusters):
    assume(obs.shape[0] >= n_clusters)
    assume(obs.shape[1] > 0)

    try:
        Z = hierarchy.linkage(obs, method='ward')
        clusters = hierarchy.fcluster(Z, n_clusters, criterion='maxclust')

        unique_clusters = len(np.unique(clusters))
        assert unique_clusters == n_clusters,             f"Expected {n_clusters} clusters, got {unique_clusters}"
    except Exception as e:
        if "Must have n>=2 objects" in str(e):
            assume(False)
        raise
```

**Failing input**: `obs=array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])`, `n_clusters=2`

## Reproducing the Bug

```python
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

obs = np.array([[0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.]])

Z = linkage(obs, method='ward')
clusters = fcluster(Z, 2, criterion='maxclust')

print(f"Requested: 2 clusters")
print(f"Got: {len(np.unique(clusters))} cluster(s)")
print(f"Cluster assignments: {clusters}")
```

**Output:**
```
Requested: 2 clusters
Got: 1 cluster(s)
Cluster assignments: [1 1 1]
```

**Extended reproduction showing the pattern:**
```python
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

obs = np.array([[0., 0.], [0., 0.], [0., 0.], [0., 0.]])
Z = linkage(obs, method='ward')

for k in [1, 2, 3, 4]:
    clusters = fcluster(Z, k, criterion='maxclust')
    n_actual = len(np.unique(clusters))
    print(f"k={k}: got {n_actual} clusters (expected {k})")
```

**Output:**
```
k=1: got 1 clusters (expected 1) ✓
k=2: got 1 clusters (expected 2) ✗
k=3: got 1 clusters (expected 3) ✗
k=4: got 4 clusters (expected 4) ✓
```

## Why This Is A Bug

1. **Violates API contract**: The documentation states that with `criterion='maxclust'`, the parameter `t` is "max number of clusters requested", but users reasonably expect to get exactly k clusters (or at least a warning if that's impossible).

2. **Inconsistent behavior**: The function returns exactly k clusters for k=1 and k=n (number of observations), but returns 1 cluster for all intermediate values. This inconsistency is confusing and unpredictable.

3. **Silent failure**: The function silently returns the wrong number of clusters without any warning or error, which can lead to incorrect data analysis results.

4. **Affects all linkage methods**: This bug affects all linkage methods (single, complete, average, weighted, centroid, median, ward), not just a specific one.

5. **Works with tiny perturbations**: When observations have tiny differences (e.g., 1e-15), the function works correctly and returns exactly k clusters. This suggests the edge case of exactly zero distances is not being handled properly.

The root cause is that when all merge distances are exactly 0, there is no threshold value that can produce an intermediate number of clusters (only 1 cluster for threshold >= 0, or n clusters for threshold < 0). The `maxclust` algorithm fails to handle this degenerate case.

## Fix

The function should handle the zero-distance edge case explicitly. When it's impossible to find a threshold that produces exactly k clusters, the function should either:

1. Arbitrarily split the identical observations into k clusters
2. Raise a warning and return the closest achievable number of clusters
3. Raise an error indicating that the requested clustering is not possible

A high-level fix would be to detect when all remaining merge distances are zero and handle this case specially in the fcluster implementation. The fix would need to be made in the Cython code that implements fcluster's maxclust logic.

Since this requires modifications to Cython code and understanding the internal algorithm, a detailed patch is beyond the scope of this report, but the issue location would be in the `fcluster` function's handling of the 'maxclust' criterion when searching for the optimal threshold.
