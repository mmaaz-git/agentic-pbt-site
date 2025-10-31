# Bug Report: scipy.spatial.distance.braycurtis Returns NaN for All-Zero Vectors

**Target**: `scipy.spatial.distance.braycurtis`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `braycurtis` distance function returns `nan` when computing the distance between identical all-zero vectors, violating the fundamental identity property that distance(x, x) should equal 0 for any valid input.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings
from scipy.spatial.distance import braycurtis


@given(
    st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=1, max_size=100)
)
@settings(max_examples=500)
def test_braycurtis_identity(u_list):
    u = np.array(u_list)
    d = braycurtis(u, u)
    assert np.isclose(d, 0.0), f"braycurtis(u, u) should be 0, got {d}"

if __name__ == "__main__":
    test_braycurtis_identity()
```

<details>

<summary>
**Failing input**: `u_list=[0.0]`
</summary>
```
/home/npc/.local/lib/python3.13/site-packages/scipy/spatial/distance.py:1245: RuntimeWarning: invalid value encountered in scalar divide
  return l1_diff.sum() / l1_sum.sum()
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/55/hypo.py", line 16, in <module>
    test_braycurtis_identity()
    ~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/55/hypo.py", line 7, in test_braycurtis_identity
    st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=1, max_size=100)
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/55/hypo.py", line 13, in test_braycurtis_identity
    assert np.isclose(d, 0.0), f"braycurtis(u, u) should be 0, got {d}"
           ~~~~~~~~~~^^^^^^^^
AssertionError: braycurtis(u, u) should be 0, got nan
Falsifying example: test_braycurtis_identity(
    u_list=[0.0],
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
from scipy.spatial.distance import braycurtis

# Test case 1: All-zero vectors (the failing case)
u = np.array([0.0, 0.0, 0.0])
v = np.array([0.0, 0.0, 0.0])
result = braycurtis(u, v)
print(f"braycurtis([0, 0, 0], [0, 0, 0]) = {result}")
print(f"Is result nan? {np.isnan(result)}")
print()

# Test case 2: Identical non-zero vectors (should return 0)
u2 = np.array([1.0, 2.0, 3.0])
v2 = np.array([1.0, 2.0, 3.0])
result2 = braycurtis(u2, v2)
print(f"braycurtis([1, 2, 3], [1, 2, 3]) = {result2}")
print(f"Is result2 close to 0? {np.isclose(result2, 0.0)}")
print()

# Test case 3: Different vectors (should return a value between 0 and 1)
u3 = np.array([1.0, 0.0, 0.0])
v3 = np.array([0.0, 1.0, 0.0])
result3 = braycurtis(u3, v3)
print(f"braycurtis([1, 0, 0], [0, 1, 0]) = {result3}")
```

<details>

<summary>
RuntimeWarning and NaN result for all-zero vectors
</summary>
```
/home/npc/.local/lib/python3.13/site-packages/scipy/spatial/distance.py:1245: RuntimeWarning: invalid value encountered in scalar divide
  return l1_diff.sum() / l1_sum.sum()
braycurtis([0, 0, 0], [0, 0, 0]) = nan
Is result nan? True

braycurtis([1, 2, 3], [1, 2, 3]) = 0.0
Is result2 close to 0? True

braycurtis([1, 0, 0], [0, 1, 0]) = 1.0
```
</details>

## Why This Is A Bug

This violates expected behavior for several critical reasons:

1. **Violates the identity property**: A fundamental property of any distance or dissimilarity measure is that d(x, x) = 0 for all valid inputs x. The braycurtis function returns `nan` instead of 0 when computing the distance between identical all-zero vectors, breaking this mathematical axiom.

2. **Mathematical interpretation**: The Bray-Curtis formula is `sum(|u_i - v_i|) / sum(|u_i + v_i|)`. When both vectors are all zeros, this becomes 0/0, which is mathematically indeterminate. However, in the context of measuring dissimilarity between identical vectors, the result should be 0 (no dissimilarity), not undefined.

3. **Documentation ambiguity**: The SciPy documentation states the function "is undefined if the inputs are of length zero" (empty arrays), but does not explicitly address the case of non-empty arrays containing all zeros. This creates confusion about expected behavior.

4. **Inconsistency with other SciPy distance functions**: The `jaccard` distance function was specifically fixed in SciPy 1.2.0 to return 0 instead of `nan` for similar edge cases. Testing confirms that `jaccard([0,0,0], [0,0,0])` returns 0.0, while `braycurtis([0,0,0], [0,0,0])` returns `nan`. This inconsistency makes the API unpredictable.

5. **Real-world impact**: In ecological applications (the original context for Bray-Curtis), all-zero vectors represent valid data points (e.g., sites with no species present). Returning `nan` for these cases can break downstream analyses that expect valid distance values.

## Relevant Context

The Bray-Curtis dissimilarity coefficient was originally developed for ecological data to quantify the compositional dissimilarity between two sites based on species counts. The function is implemented in `/home/npc/.local/lib/python3.13/site-packages/scipy/spatial/distance.py` at line 1245.

The issue occurs because the function performs division without checking if the denominator is zero:
```python
return l1_diff.sum() / l1_sum.sum()
```

Similar distance functions in SciPy handle this edge case differently:
- `jaccard`: Returns 0.0 for all-zero vectors (fixed in v1.2.0)
- `canberra`: Has explicit documentation stating "When u[i] and v[i] are 0 for given i, then the fraction 0/0 = 0 is used"
- `euclidean` and `cityblock`: Return 0.0 for identical all-zero vectors

Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.braycurtis.html
Source code: https://github.com/scipy/scipy/blob/main/scipy/spatial/distance.py

## Proposed Fix

Add a check for zero denominator before division to maintain consistency with other distance functions and satisfy the identity property:

```diff
--- a/scipy/spatial/distance.py
+++ b/scipy/spatial/distance.py
@@ -1242,7 +1242,11 @@ def braycurtis(u, v, w=None):
         w = _validate_weights(w)
         l1_diff = w * l1_diff
         l1_sum = w * l1_sum
-    return l1_diff.sum() / l1_sum.sum()
+    denominator = l1_sum.sum()
+    if denominator == 0:
+        return 0.0
+    else:
+        return l1_diff.sum() / denominator
```