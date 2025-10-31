# Bug Report: scipy.spatial.distance.dice Returns NaN for All-False Boolean Arrays

**Target**: `scipy.spatial.distance.dice`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `dice` function returns NaN when both input boolean arrays contain only False values, instead of returning 0.0 for identical inputs as expected for a distance metric.

## Property-Based Test

```python
import math
import numpy as np
from hypothesis import given, settings, strategies as st
from scipy.spatial import distance

@settings(max_examples=1000)
@given(n=st.integers(1, 20))
def test_dice_all_zeros(n):
    x = np.zeros(n, dtype=bool)
    y = np.zeros(n, dtype=bool)

    dist = distance.dice(x, y)

    assert math.isclose(dist, 0.0, abs_tol=1e-9)

# Run the test
if __name__ == "__main__":
    test_dice_all_zeros()
```

<details>

<summary>
**Failing input**: `n=1`
</summary>
```
/home/npc/.local/lib/python3.13/site-packages/scipy/spatial/distance.py:1503: RuntimeWarning: invalid value encountered in divide
  return float((ntf + nft) / np.array(2.0 * ntt + ntf + nft))
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/57/hypo.py", line 18, in <module>
    test_dice_all_zeros()
    ~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/57/hypo.py", line 7, in test_dice_all_zeros
    @given(n=st.integers(1, 20))
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/57/hypo.py", line 14, in test_dice_all_zeros
    assert math.isclose(dist, 0.0, abs_tol=1e-9)
           ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_dice_all_zeros(
    n=1,
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
from scipy.spatial import distance

# Test case: all-False boolean arrays
x = np.array([False, False, False])
y = np.array([False, False, False])

# Compute dice dissimilarity
result = distance.dice(x, y)

print(f"dice({x}, {y}) = {result}")
print(f"Expected: 0.0 (identical arrays should have distance 0)")
print(f"Actual: {result}")
print()

# Comparison with other similar functions
print("Comparison with other boolean distance metrics:")
print(f"jaccard({x}, {y}) = {distance.jaccard(x, y)}")
print(f"hamming({x}, {y}) = {distance.hamming(x, y)}")
print(f"rogerstanimoto({x}, {y}) = {distance.rogerstanimoto(x, y)}")
```

<details>

<summary>
RuntimeWarning: invalid value encountered in divide, returns nan instead of 0.0
</summary>
```
/home/npc/.local/lib/python3.13/site-packages/scipy/spatial/distance.py:1503: RuntimeWarning: invalid value encountered in divide
  return float((ntf + nft) / np.array(2.0 * ntt + ntf + nft))
dice([False False False], [False False False]) = nan
Expected: 0.0 (identical arrays should have distance 0)
Actual: nan

Comparison with other boolean distance metrics:
jaccard([False False False], [False False False]) = 0.0
hamming([False False False], [False False False]) = 0.0
rogerstanimoto([False False False], [False False False]) = 0.0
```
</details>

## Why This Is A Bug

This violates expected behavior for several critical reasons:

1. **Mathematical Principle Violation**: A fundamental property of any distance/dissimilarity metric is that the distance between two identical vectors should be 0. The Dice dissimilarity between two identical arrays `[False, False, False]` should be 0.0, not NaN.

2. **Documented Historical Precedent**: The `scipy.spatial.distance.jaccard` function had this exact same bug, which was explicitly fixed in SciPy v1.2.0. The jaccard documentation states (lines 858-861 in distance.py):
   ```
   .. versionchanged:: 1.2.0
      Previously, if all (positively weighted) elements in `u` and `v` are
      zero, the function would return ``nan``.  This was changed to return
      ``0`` instead.
   ```

3. **Inconsistency with Similar Functions**: All other boolean distance metrics in the same module handle this edge case correctly:
   - `jaccard([False, False, False], [False, False, False])` returns 0.0
   - `hamming([False, False, False], [False, False, False])` returns 0.0
   - `rogerstanimoto([False, False, False], [False, False, False])` returns 0.0

4. **Runtime Warning**: The function produces a "RuntimeWarning: invalid value encountered in divide" warning, indicating problematic behavior that was not intentionally designed.

5. **Real-World Impact**: Users working with sparse boolean data commonly encounter all-False arrays. Examples include:
   - Document term vectors where no terms are present
   - Biological feature vectors where no features are expressed
   - Set operations on empty sets
   The NaN value propagates through downstream computations, potentially breaking entire data analysis pipelines.

## Relevant Context

The bug occurs at line 1503 in `/home/npc/pbt/agentic-pbt/envs/scipy_env/lib/python3.13/site-packages/scipy/spatial/distance.py`:

```python
return float((ntf + nft) / np.array(2.0 * ntt + ntf + nft))
```

When both input arrays contain only False values:
- `ntt` (True-True count) = 0
- `ntf` (True-False count) = 0
- `nft` (False-True count) = 0

This results in division by zero (0/0), which produces NaN.

The Dice coefficient formula from the documentation is:
```
(c_TF + c_FT) / (2*c_TT + c_FT + c_TF)
```

For all-False arrays, all counts are 0, resulting in 0/0 = NaN.

The jaccard function at line 816 explicitly documents: "If u and v are both zero, their Jaccard dissimilarity is defined to be zero."

Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.dice.html
Source code: https://github.com/scipy/scipy/blob/main/scipy/spatial/distance.py#L1441

## Proposed Fix

```diff
--- a/scipy/spatial/distance.py
+++ b/scipy/spatial/distance.py
@@ -1500,7 +1500,10 @@ def dice(u, v, w=None):
         else:
             ntt = (u * v * w).sum()
     (nft, ntf) = _nbool_correspond_ft_tf(u, v, w=w)
-    return float((ntf + nft) / np.array(2.0 * ntt + ntf + nft))
+    denominator = 2.0 * ntt + ntf + nft
+    if denominator == 0:
+        return 0.0
+    return float((ntf + nft) / np.array(denominator))


 def rogerstanimoto(u, v, w=None):
```