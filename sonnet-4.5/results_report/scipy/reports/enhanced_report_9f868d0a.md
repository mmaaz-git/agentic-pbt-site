# Bug Report: scipy.spatial.distance.dice Returns NaN for All-False Arrays

**Target**: `scipy.spatial.distance.dice`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `dice` dissimilarity function returns `nan` when both input boolean arrays contain only `False` values, violating the fundamental metric property that the distance between identical vectors should be 0.

## Property-Based Test

```python
from scipy.spatial.distance import dice
from hypothesis import given, strategies as st, assume
import numpy as np


@given(
    st.lists(st.integers(min_value=0, max_value=1), min_size=1, max_size=30),
    st.lists(st.integers(min_value=0, max_value=1), min_size=1, max_size=30)
)
def test_dice_distance_range(u, v):
    assume(len(u) == len(v))
    u_arr = np.array(u, dtype=bool)
    v_arr = np.array(v, dtype=bool)
    d = dice(u_arr, v_arr)
    assert 0.0 <= d <= 1.0 + 1e-9


if __name__ == "__main__":
    test_dice_distance_range()
```

<details>

<summary>
**Failing input**: `u=[0], v=[0]`
</summary>
```
/home/npc/.local/lib/python3.13/site-packages/scipy/spatial/distance.py:1503: RuntimeWarning: invalid value encountered in divide
  return float((ntf + nft) / np.array(2.0 * ntt + ntf + nft))
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/26/hypo.py", line 19, in <module>
    test_dice_distance_range()
    ~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/26/hypo.py", line 7, in test_dice_distance_range
    st.lists(st.integers(min_value=0, max_value=1), min_size=1, max_size=30),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/26/hypo.py", line 15, in test_dice_distance_range
    assert 0.0 <= d <= 1.0 + 1e-9
           ^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_dice_distance_range(
    u=[0],
    v=[0],
)
```
</details>

## Reproducing the Bug

```python
from scipy.spatial.distance import dice
import numpy as np

u = np.array([False, False, False])
v = np.array([False, False, False])

result = dice(u, v)
print(f"dice([False, False, False], [False, False, False]) = {result}")

assert result == 0.0, f"Expected 0.0, got {result}"
```

<details>

<summary>
RuntimeWarning and AssertionError when computing dice dissimilarity for identical all-False arrays
</summary>
```
/home/npc/.local/lib/python3.13/site-packages/scipy/spatial/distance.py:1503: RuntimeWarning: invalid value encountered in divide
  return float((ntf + nft) / np.array(2.0 * ntt + ntf + nft))
dice([False, False, False], [False, False, False]) = nan
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/26/repo.py", line 10, in <module>
    assert result == 0.0, f"Expected 0.0, got {result}"
           ^^^^^^^^^^^^^
AssertionError: Expected 0.0, got nan
```
</details>

## Why This Is A Bug

The Dice dissimilarity metric violates fundamental mathematical properties when both input arrays are all-False:

1. **Identity Property Violation**: Any dissimilarity metric must satisfy d(x, x) = 0 for identical vectors. The function returns `nan` instead of 0.0 for identical all-False arrays.

2. **Invalid Return Value**: The documentation states that `dice` returns "The Dice dissimilarity between 1-D arrays u and v" as a double. A dissimilarity metric should return a value in [0, 1], but `nan` is outside this valid range.

3. **Mathematical Cause**: The formula (c_TF + c_FT)/(2*c_TT + c_FT + c_TF) results in 0/0 when all values are False because:
   - c_TT = 0 (no True-True matches)
   - c_TF = 0 (no True-False matches)
   - c_FT = 0 (no False-True matches)
   - Denominator = 2*0 + 0 + 0 = 0

4. **Inconsistency with Other Metrics**: Other scipy dissimilarity metrics handle this case correctly:
   - `jaccard([False, False, False], [False, False, False])` returns 0.0
   - `hamming([False, False, False], [False, False, False])` returns 0.0
   - `rogerstanimoto([False, False, False], [False, False, False])` returns 0.0
   - Only `sokalsneath` explicitly raises an error stating it's undefined for all-False vectors

## Relevant Context

The Dice coefficient (also known as Sørensen-Dice coefficient) is a statistic used for gauging the similarity of two samples. The dissimilarity is computed as 1 - similarity. The current implementation is found in `/home/npc/.local/lib/python3.13/site-packages/scipy/spatial/distance.py` at line 1503.

This bug affects real-world applications dealing with sparse binary features, such as:
- Document similarity where no terms are present
- Feature vectors in machine learning where no features are active
- Set comparisons where both sets are empty

The scipy documentation for `dice` (lines 1441-1503) doesn't specify behavior for all-False vectors, but mathematical convention and consistency with other metrics strongly suggest returning 0.0 for identical vectors.

## Proposed Fix

```diff
--- a/scipy/spatial/distance.py
+++ b/scipy/spatial/distance.py
@@ -1500,4 +1500,7 @@ def dice(u, v, w=None):
         else:
             ntt = (u * v * w).sum()
     (nft, ntf) = _nbool_correspond_ft_tf(u, v, w=w)
-    return float((ntf + nft) / np.array(2.0 * ntt + ntf + nft))
+    denom = 2.0 * ntt + ntf + nft
+    if denom == 0:
+        return 0.0
+    return float((ntf + nft) / np.array(denom))
```