# Bug Report: scipy.spatial.distance.braycurtis Returns NaN for All-Zero Arrays

**Target**: `scipy.spatial.distance.braycurtis`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `braycurtis` distance function returns `nan` when both input arrays contain only zeros, violating the fundamental metric property that the distance between identical vectors should be 0.

## Property-Based Test

```python
from scipy.spatial.distance import braycurtis
from hypothesis import given, strategies as st, assume
import numpy as np


@given(
    st.lists(st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
             min_size=1, max_size=30),
    st.lists(st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
             min_size=1, max_size=30)
)
def test_braycurtis_range_positive_inputs(u, v):
    assume(len(u) == len(v))
    u_arr = np.array(u)
    v_arr = np.array(v)
    d = braycurtis(u_arr, v_arr)
    assert 0.0 <= d <= 1.0 + 1e-9

if __name__ == "__main__":
    test_braycurtis_range_positive_inputs()
```

<details>

<summary>
**Failing input**: `u=[0.0], v=[0.0]`
</summary>
```
/home/npc/.local/lib/python3.13/site-packages/scipy/spatial/distance.py:1245: RuntimeWarning: invalid value encountered in scalar divide
  return l1_diff.sum() / l1_sum.sum()
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/31/hypo.py", line 20, in <module>
    test_braycurtis_range_positive_inputs()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/31/hypo.py", line 7, in test_braycurtis_range_positive_inputs
    st.lists(st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/31/hypo.py", line 17, in test_braycurtis_range_positive_inputs
    assert 0.0 <= d <= 1.0 + 1e-9
           ^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_braycurtis_range_positive_inputs(
    u=[0.0],
    v=[0.0],
)
```
</details>

## Reproducing the Bug

```python
from scipy.spatial.distance import braycurtis
import numpy as np

# Test case 1: All-zero arrays
u = np.array([0.0, 0.0, 0.0])
v = np.array([0.0, 0.0, 0.0])

print("Testing braycurtis with all-zero arrays:")
print(f"u = {u}")
print(f"v = {v}")

result = braycurtis(u, v)
print(f"braycurtis([0, 0, 0], [0, 0, 0]) = {result}")

# Verify it should be 0 for identical vectors
print(f"\nAssertion check: result == 0.0")
try:
    assert result == 0.0, f"Expected 0.0, got {result}"
    print("Assertion passed!")
except AssertionError as e:
    print(f"Assertion failed: {e}")

# Additional test: Identical non-zero vectors for comparison
print("\n--- For comparison: identical non-zero vectors ---")
u2 = np.array([1.0, 2.0, 3.0])
v2 = np.array([1.0, 2.0, 3.0])
result2 = braycurtis(u2, v2)
print(f"braycurtis([1, 2, 3], [1, 2, 3]) = {result2}")
```

<details>

<summary>
RuntimeWarning and NaN result for all-zero arrays
</summary>
```
/home/npc/.local/lib/python3.13/site-packages/scipy/spatial/distance.py:1245: RuntimeWarning: invalid value encountered in scalar divide
  return l1_diff.sum() / l1_sum.sum()
Testing braycurtis with all-zero arrays:
u = [0. 0. 0.]
v = [0. 0. 0.]
braycurtis([0, 0, 0], [0, 0, 0]) = nan

Assertion check: result == 0.0
Assertion failed: Expected 0.0, got nan

--- For comparison: identical non-zero vectors ---
braycurtis([1, 2, 3], [1, 2, 3]) = 0.0
```
</details>

## Why This Is A Bug

The Bray-Curtis distance violates a fundamental property of distance metrics: d(x, x) = 0 for any vector x, including the zero vector.

The mathematical formula is: Σ|u_i - v_i| / Σ|u_i + v_i|

When u = v = [0, 0, ..., 0]:
- Numerator: Σ|0 - 0| = 0 (correct)
- Denominator: Σ|0 + 0| = 0 (causes division by zero)
- Result: 0/0 → NaN (incorrect, should be 0)

The documentation states the function "is undefined if the inputs are of length zero" but this is misleading:
1. The actual issue occurs when all values are zero, not when arrays have zero length
2. The function accepts arrays of any positive length, including [0.0]
3. The documentation doesn't address the all-zeros case explicitly

The function correctly returns 0 for identical non-zero vectors (e.g., [1,2,3] compared to itself), demonstrating inconsistent behavior that only manifests with zero vectors.

## Relevant Context

The Bray-Curtis dissimilarity is widely used in ecology to quantify compositional dissimilarity between sites based on species counts. Zero vectors represent empty sites (no species present), which are valid ecological data points.

The ecological literature acknowledges this issue and has developed "zero-adjusted Bray-Curtis" methods to handle it. The current SciPy implementation generates a RuntimeWarning and returns NaN, which propagates through subsequent calculations and can break analysis pipelines.

Relevant code location: `/home/npc/pbt/agentic-pbt/envs/scipy_env/lib/python3.13/site-packages/scipy/spatial/distance.py:1245`

Similar distance functions like Canberra distance (see lines 1276-1277 in the same file) explicitly handle the 0/0 case by treating it as 0, providing a precedent for fixing this issue.

## Proposed Fix

```diff
--- a/scipy/spatial/distance.py
+++ b/scipy/spatial/distance.py
@@ -1242,4 +1242,7 @@ def braycurtis(u, v, w=None):
         w = _validate_weights(w)
         l1_diff = w * l1_diff
         l1_sum = w * l1_sum
-    return l1_diff.sum() / l1_sum.sum()
+    denom = l1_sum.sum()
+    if denom == 0:
+        return 0.0
+    return l1_diff.sum() / denom
```