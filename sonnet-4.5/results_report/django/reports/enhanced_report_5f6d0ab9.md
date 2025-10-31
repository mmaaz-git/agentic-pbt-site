# Bug Report: scipy.spatial.distance.correlation Returns NaN for Constant Arrays Violating Metric Identity Property

**Target**: `scipy.spatial.distance.correlation`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `correlation` distance function returns `nan` when given constant arrays (arrays where all elements have the same value), violating the fundamental metric property that the distance between identical vectors must be 0.

## Property-Based Test

```python
from scipy.spatial.distance import correlation
from hypothesis import given, strategies as st, assume
import numpy as np


@given(
    st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
             min_size=2, max_size=20)
)
def test_correlation_identity(u):
    """Property: correlation(u, u) should be 0"""
    u_arr = np.array(u)
    d = correlation(u_arr, u_arr)
    assert np.isclose(d, 0.0, atol=1e-9) or not np.isnan(d)
```

<details>

<summary>
**Failing input**: `u=[0.0, 0.0]`
</summary>
```
/home/npc/.local/lib/python3.13/site-packages/scipy/spatial/distance.py:682: RuntimeWarning: invalid value encountered in scalar divide
  dist = 1.0 - uv / math.sqrt(uu * vv)
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 18, in <module>
    test_correlation_identity()
    ~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 7, in test_correlation_identity
    st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 14, in test_correlation_identity
    assert np.isclose(d, 0.0, atol=1e-9) or not np.isnan(d)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_correlation_identity(
    u=[0.0, 0.0],
)
```
</details>

## Reproducing the Bug

```python
from scipy.spatial.distance import correlation
import numpy as np

u = np.array([5.0, 5.0, 5.0])
v = np.array([5.0, 5.0, 5.0])

result = correlation(u, v)

print(f"correlation([5, 5, 5], [5, 5, 5]) = {result}")
assert result == 0.0, f"Expected 0.0 for identical arrays, got {result}"
```

<details>

<summary>
AssertionError: Expected 0.0 for identical arrays, got nan
</summary>
```
/home/npc/.local/lib/python3.13/site-packages/scipy/spatial/distance.py:682: RuntimeWarning: invalid value encountered in scalar divide
  dist = 1.0 - uv / math.sqrt(uu * vv)
correlation([5, 5, 5], [5, 5, 5]) = nan
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/50/repo.py", line 10, in <module>
    assert result == 0.0, f"Expected 0.0 for identical arrays, got {result}"
           ^^^^^^^^^^^^^
AssertionError: Expected 0.0 for identical arrays, got nan
```
</details>

## Why This Is A Bug

This violates the fundamental identity axiom of distance metrics: d(x, x) = 0 for any vector x. The scipy documentation states that `correlation` computes "the correlation distance between two 1-D arrays" using the formula:

```
1 - [(u - ū) · (v - v̄)] / [||u - ū||₂ * ||v - v̄||₂]
```

For constant arrays where all elements are the same value, the variance is zero (ū = u[i] for all i), making both ||u - ū||₂ = 0 and ||v - v̄||₂ = 0. This causes division by zero on line 682 of distance.py:

```python
dist = 1.0 - uv / math.sqrt(uu * vv)  # uu = 0, vv = 0 when arrays are constant
```

The result is `nan` with a RuntimeWarning about "invalid value encountered in scalar divide". This breaks:
1. The mathematical definition of a proper distance metric
2. Property-based testing frameworks that generate constant arrays
3. Any downstream code expecting metric properties to hold
4. Consistency with other scipy distance functions (e.g., `euclidean([5,5,5], [5,5,5])` correctly returns 0.0)

The documentation does not mention this edge case or warn users about `nan` returns for constant arrays.

## Relevant Context

- The issue occurs in scipy.spatial.distance.py at line 682
- Affects all constant arrays regardless of value or length (tested with [0,0], [5,5,5], [-5,-5,-5,-5], [100]*10)
- Non-constant arrays work correctly: `correlation([1,2,3], [1,2,3])` returns 0.0
- The scipy version tested is 1.16.2
- Mathematical reference: Distance metrics must satisfy the identity property d(x,x) = 0 (see [Metric space axioms](https://en.wikipedia.org/wiki/Metric_space#Definition))
- Source code location: `/home/npc/.local/lib/python3.13/site-packages/scipy/spatial/distance.py`

## Proposed Fix

```diff
--- a/scipy/spatial/distance.py
+++ b/scipy/spatial/distance.py
@@ -679,6 +679,11 @@ def correlation(u, v, w=None, centered=True):
     uv = np.dot(u, vw)
     uu = np.dot(u, uw)
     vv = np.dot(v, vw)
-    dist = 1.0 - uv / math.sqrt(uu * vv)
+    denom = math.sqrt(uu * vv)
+    if denom == 0:
+        # For constant arrays, return 0 if identical (identity property), nan otherwise
+        return 0.0 if np.allclose(u, v) else np.nan
+    dist = 1.0 - uv / denom
     # Clip the result to avoid rounding error
     return np.clip(dist, 0.0, 2.0)
```