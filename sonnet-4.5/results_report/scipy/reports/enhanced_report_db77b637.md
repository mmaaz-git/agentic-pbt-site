# Bug Report: scipy.spatial.distance.dice Returns NaN for All-False Boolean Arrays

**Target**: `scipy.spatial.distance.dice`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `dice` distance function returns NaN when both input boolean arrays contain only False values, violating the fundamental distance metric property that identical inputs should have distance 0.

## Property-Based Test

```python
import math

import numpy as np
from hypothesis import given, settings, strategies as st
from hypothesis.extra import numpy as npst
from scipy.spatial import distance


@given(
    u=npst.arrays(
        dtype=np.bool_,
        shape=st.integers(min_value=1, max_value=100),
    ),
    v=npst.arrays(
        dtype=np.bool_,
        shape=st.integers(min_value=1, max_value=100),
    ),
)
@settings(max_examples=300)
def test_dice_symmetry(u, v):
    if u.shape != v.shape:
        return

    d_uv = distance.dice(u, v)
    d_vu = distance.dice(v, u)

    assert math.isclose(d_uv, d_vu, rel_tol=1e-9, abs_tol=1e-9)


if __name__ == "__main__":
    test_dice_symmetry()
```

<details>

<summary>
**Failing input**: `u=array([False]), v=array([False])`
</summary>
```
/home/npc/.local/lib/python3.13/site-packages/scipy/spatial/distance.py:1503: RuntimeWarning: invalid value encountered in divide
  return float((ntf + nft) / np.array(2.0 * ntt + ntf + nft))
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 31, in <module>
    test_dice_symmetry()
    ~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 10, in test_dice_symmetry
    u=npst.arrays(
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 27, in test_dice_symmetry
    assert math.isclose(d_uv, d_vu, rel_tol=1e-9, abs_tol=1e-9)
           ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_dice_symmetry(
    u=array([False]),
    v=array([False]),
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/25/hypo.py:24
```
</details>

## Reproducing the Bug

```python
import numpy as np
from scipy.spatial import distance

# Test case: both arrays contain only False values
u = np.array([False])
v = np.array([False])

print("Testing dice distance with all-False arrays:")
print(f"u = {u}")
print(f"v = {v}")
print()

result = distance.dice(u, v)
print(f"dice([False], [False]) = {result}")
print()

# Test with larger all-False arrays
u_large = np.array([False, False, False])
v_large = np.array([False, False, False])
result_large = distance.dice(u_large, v_large)
print(f"dice([False, False, False], [False, False, False]) = {result_large}")
print()

# For comparison with other distance functions
print("Comparison with other distance metrics:")
print(f"jaccard([False], [False]) = {distance.jaccard(u, v)}")
print(f"hamming([False], [False]) = {distance.hamming(u, v)}")
print()

# Demonstrate the NaN equality issue
d_uv = distance.dice(u, v)
d_vu = distance.dice(v, u)
print("Symmetry check:")
print(f"dice(u, v) = {d_uv}")
print(f"dice(v, u) = {d_vu}")
print(f"dice(u, v) == dice(v, u): {d_uv == d_vu}")
print(f"Both are NaN: {np.isnan(d_uv) and np.isnan(d_vu)}")
```

<details>

<summary>
RuntimeWarning: invalid value encountered in divide
</summary>
```
/home/npc/.local/lib/python3.13/site-packages/scipy/spatial/distance.py:1503: RuntimeWarning: invalid value encountered in divide
  return float((ntf + nft) / np.array(2.0 * ntt + ntf + nft))
Testing dice distance with all-False arrays:
u = [False]
v = [False]

dice([False], [False]) = nan

dice([False, False, False], [False, False, False]) = nan

Comparison with other distance metrics:
jaccard([False], [False]) = 0.0
hamming([False], [False]) = 0.0

Symmetry check:
dice(u, v) = nan
dice(v, u) = nan
dice(u, v) == dice(v, u): False
Both are NaN: True
```
</details>

## Why This Is A Bug

This bug violates fundamental distance metric properties and expected behavior in multiple ways:

1. **Violates the identity property of distance metrics**: For any distance metric d, the property d(x,x) = 0 must hold. Since `[False]` and `[False]` are identical arrays, `dice([False], [False])` should return 0, not NaN.

2. **Breaks symmetry testing**: While mathematically `dice(u,v)` and `dice(v,u)` both return NaN, the property-based test correctly identifies that `NaN == NaN` evaluates to False in Python, which breaks equality comparisons and symmetry verification.

3. **Inconsistent with similar metrics in the same module**: The `jaccard` and `hamming` distance functions correctly return 0.0 for all-False arrays. In fact, scipy specifically fixed this exact issue for `jaccard` in version 1.2.0, establishing clear precedent.

4. **Undocumented behavior**: The documentation for `dice` does not mention that NaN can be returned, nor does it specify any preconditions that would exclude all-False inputs. Users have no warning about this edge case.

5. **Mathematical formula produces 0/0**: The Dice coefficient formula `(c_TF + c_FT) / (2*c_TT + c_TF + c_FT)` evaluates to 0/0 when all values are False (c_TT = c_TF = c_FT = 0), which is undefined and produces NaN.

6. **NaN propagation causes downstream issues**: NaN values propagate through numerical computations, potentially breaking entire pipelines that use this distance metric without explicit NaN handling.

## Relevant Context

The issue occurs in `/home/npc/.local/lib/python3.13/site-packages/scipy/spatial/distance.py:1503` where the Dice formula is computed without checking for division by zero.

The scipy library has already established precedent for handling this exact edge case. According to the scipy 1.2.0 changelog for the `jaccard` function: "Previously, if all (positively weighted) elements in u and v are zero, the function would return nan. This was changed to return 0 instead."

The implementation logic shows that when both arrays are all False:
- `ntt = (u & v).sum() = 0` (no True-True pairs)
- `nft = (~u & v).sum() = 0` (False-True pairs)
- `ntf = (u & ~v).sum() = 0` (True-False pairs)
- The denominator becomes `2*0 + 0 + 0 = 0`, causing division by zero

Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.dice.html
Source code: https://github.com/scipy/scipy/blob/main/scipy/spatial/distance.py

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
```