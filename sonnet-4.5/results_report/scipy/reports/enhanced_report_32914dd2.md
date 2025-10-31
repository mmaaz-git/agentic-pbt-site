# Bug Report: scipy.spatial.distance.braycurtis Division by Zero with All-Zero Vectors

**Target**: `scipy.spatial.distance.braycurtis`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `braycurtis` distance function returns `nan` when computing the distance between all-zero vectors due to division by zero, violating the fundamental identity property that the distance between identical vectors should be 0.

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
  File "/home/npc/pbt/agentic-pbt/worker_/15/hypo.py", line 17, in <module>
    test_braycurtis_identity()
    ~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/15/hypo.py", line 7, in test_braycurtis_identity
    st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=1, max_size=100)
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/15/hypo.py", line 13, in test_braycurtis_identity
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

# Test case: all-zero vectors
u = np.array([0.0, 0.0, 0.0])
v = np.array([0.0, 0.0, 0.0])

print(f"Input vectors:")
print(f"u = {u}")
print(f"v = {v}")
print()

# Compute braycurtis distance
result = braycurtis(u, v)

print(f"braycurtis(u, v) = {result}")
print(f"Is result NaN? {np.isnan(result)}")
print()

# This should be 0 for identical vectors (identity property)
# But returns NaN due to division by zero
print("Expected: 0.0 (identity property: distance between identical vectors should be 0)")
print(f"Actual: {result}")
```

<details>

<summary>
RuntimeWarning: invalid value encountered in scalar divide
</summary>
```
/home/npc/.local/lib/python3.13/site-packages/scipy/spatial/distance.py:1245: RuntimeWarning: invalid value encountered in scalar divide
  return l1_diff.sum() / l1_sum.sum()
Input vectors:
u = [0. 0. 0.]
v = [0. 0. 0.]

braycurtis(u, v) = nan
Is result NaN? True

Expected: 0.0 (identity property: distance between identical vectors should be 0)
Actual: nan
```
</details>

## Why This Is A Bug

1. **Violates the identity property**: A fundamental axiom of dissimilarity measures is that d(x, x) = 0 for any vector x. The Bray-Curtis distance should return 0 for identical vectors, but returns `nan` when both vectors are all zeros.

2. **Mathematical interpretation**: The Bray-Curtis formula is Σ|u_i - v_i| / Σ|u_i + v_i|. For identical all-zero vectors, this gives 0/0. While mathematically undefined, in the context of distance metrics, the limiting behavior and semantic meaning suggest this should be 0 (no distance between identical points).

3. **Inconsistent with similar functions in scipy**: The `jaccard` function (line 915 in distance.py) explicitly handles the zero denominator case with `return (a / b) if b != 0 else np.float64(0)`. The `canberra` function documents that "When u[i] and v[i] are 0 for given i, then the fraction 0/0 = 0 is used in the calculation" and uses `np.nansum` to handle NaN values gracefully.

4. **Real-world impact**: All-zero vectors commonly occur in sparse data representations, zero-padding scenarios, and initialization states. Applications using Bray-Curtis distance can unexpectedly receive NaN values that propagate through calculations.

5. **Documentation mismatch**: While the documentation states "is undefined if the inputs are of length zero", it doesn't mention the all-zero vector case, which is different from zero-length inputs.

## Relevant Context

The Bray-Curtis distance implementation is located at `/home/npc/pbt/agentic-pbt/envs/scipy_env/lib/python3.13/site-packages/scipy/spatial/distance.py:1200-1245`.

Similar distance functions in the same file handle edge cases differently:
- **jaccard** (line 796-915): Explicitly returns 0 when denominator is 0, with a note in version 1.2.0 changelog stating "Previously, if all (positively weighted) elements in u and v are zero, the function would return nan. This was changed to return 0 instead."
- **canberra** (line 1248-1300): Uses `np.errstate(invalid='ignore')` and `np.nansum` to treat 0/0 as 0.

Documentation links:
- SciPy distance functions: https://docs.scipy.org/doc/scipy/reference/spatial.distance.html
- Bray-Curtis dissimilarity: https://en.wikipedia.org/wiki/Bray%E2%80%93Curtis_dissimilarity

## Proposed Fix

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
+        return np.float64(0)
+    else:
+        return l1_diff.sum() / denominator
```