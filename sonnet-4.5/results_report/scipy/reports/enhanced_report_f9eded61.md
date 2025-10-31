# Bug Report: scipy.spatial.distance.dice Returns NaN for All-False Vectors

**Target**: `scipy.spatial.distance.dice`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `dice` function returns NaN when comparing two all-False boolean vectors, violating the fundamental metric property that d(x, x) = 0 for identical vectors.

## Property-Based Test

```python
import numpy as np
import scipy.spatial.distance as distance
from hypothesis import given, strategies as st, assume


@given(
    st.lists(st.integers(min_value=0, max_value=1), min_size=5, max_size=50),
    st.lists(st.integers(min_value=0, max_value=1), min_size=5, max_size=50)
)
def test_dice_bounds(u, v):
    assume(len(u) == len(v))
    u_arr = np.array(u, dtype=bool)
    v_arr = np.array(v, dtype=bool)

    d = distance.dice(u_arr, v_arr)

    assert 0 <= d <= 1, f"Dice dissimilarity should be in [0,1], got {d}"


if __name__ == "__main__":
    # Run the property-based test
    test_dice_bounds()
```

<details>

<summary>
**Failing input**: `u=[0, 0, 0, 0, 0], v=[0, 0, 0, 0, 0]`
</summary>
```
/home/npc/.local/lib/python3.13/site-packages/scipy/spatial/distance.py:1503: RuntimeWarning: invalid value encountered in divide
  return float((ntf + nft) / np.array(2.0 * ntt + ntf + nft))
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 22, in <module>
    test_dice_bounds()
    ~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 7, in test_dice_bounds
    st.lists(st.integers(min_value=0, max_value=1), min_size=5, max_size=50),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 17, in test_dice_bounds
    assert 0 <= d <= 1, f"Dice dissimilarity should be in [0,1], got {d}"
           ^^^^^^^^^^^
AssertionError: Dice dissimilarity should be in [0,1], got nan
Falsifying example: test_dice_bounds(
    u=[0, 0, 0, 0, 0],
    v=[0, 0, 0, 0, 0],
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
from scipy.spatial.distance import dice, jaccard

# Create two all-False boolean vectors
u = np.array([False, False, False, False, False])
v = np.array([False, False, False, False, False])

print("Testing dice function with all-False vectors:")
print(f"u = {u}")
print(f"v = {v}")
print()

# Test dice function
result = dice(u, v)
print(f"dice(all-False, all-False) = {result}")
print(f"Expected: 0.0 (since d(x,x) should be 0 for identical vectors)")
print()

# Compare with jaccard for reference
jaccard_result = jaccard(u, v)
print(f"For comparison, jaccard(all-False, all-False) = {jaccard_result}")
print("Note: jaccard was fixed in scipy 1.2.0 to return 0.0 for this case")
```

<details>

<summary>
RuntimeWarning: invalid value encountered in divide - Returns NaN instead of 0.0
</summary>
```
/home/npc/.local/lib/python3.13/site-packages/scipy/spatial/distance.py:1503: RuntimeWarning: invalid value encountered in divide
  return float((ntf + nft) / np.array(2.0 * ntt + ntf + nft))
Testing dice function with all-False vectors:
u = [False False False False False]
v = [False False False False False]

dice(all-False, all-False) = nan
Expected: 0.0 (since d(x,x) should be 0 for identical vectors)

For comparison, jaccard(all-False, all-False) = 0.0
Note: jaccard was fixed in scipy 1.2.0 to return 0.0 for this case
```
</details>

## Why This Is A Bug

1. **Violates fundamental metric property**: Any dissimilarity/distance metric must satisfy the identity axiom d(x, x) = 0. The `dice` function returns NaN for identical all-False vectors, breaking this mathematical requirement.

2. **Mathematical incorrectness**: The Dice dissimilarity formula (c_TF + c_FT) / (2*c_TT + c_TF + c_FT) produces 0/0 when both vectors are all-False (c_TT = 0, c_TF = 0, c_FT = 0). Mathematically, for identical empty sets, the dissimilarity should be 0, not undefined.

3. **Inconsistent with documented precedent**: The `jaccard` function had the exact same bug and was fixed in scipy 1.2.0. The jaccard documentation explicitly states: "Previously, if all (positively weighted) elements in u and v are zero, the function would return nan. This was changed to return 0 instead." (line 859-861 in distance.py)

4. **Inconsistent behavior within scipy**: The `jaccard` function correctly returns 0.0 for all-False vectors after its fix, while `dice` still returns NaN. Both functions have similar mathematical structures and should handle this edge case consistently.

5. **Breaks user expectations**: Functions in `scipy.spatial.distance` are expected to behave as valid distance/dissimilarity metrics. The documentation examples for `jaccard` explicitly show that `distance.jaccard([0, 0, 0], [0, 0, 0])` returns 0.0 (line 893-894).

6. **Real-world impact**: Comparing empty feature sets or sparse boolean vectors is common in document similarity, set operations, and feature matching applications. This bug causes runtime warnings and invalid results for legitimate use cases.

## Relevant Context

The dice function implementation is located at `/home/npc/.local/lib/python3.13/site-packages/scipy/spatial/distance.py:1441-1503`. The problematic line 1503 performs the division without checking for zero denominator:

```python
return float((ntf + nft) / np.array(2.0 * ntt + ntf + nft))
```

The jaccard function at line 915 in the same file demonstrates the correct approach:

```python
return (a / b) if b != 0 else np.float64(0)
```

SciPy documentation for jaccard: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jaccard.html

The Dice coefficient (also known as Sørensen-Dice coefficient) is a statistic used for gauging the similarity of two samples. When both samples are empty (all-False), they are identical and should have dissimilarity of 0.

## Proposed Fix

Apply the same fix that was applied to `jaccard` in scipy 1.2.0 to check for division by zero:

```diff
--- a/scipy/spatial/distance.py
+++ b/scipy/spatial/distance.py
@@ -1500,7 +1500,9 @@ def dice(u, v, w=None):
         else:
             ntt = (u * v * w).sum()
     (nft, ntf) = _nbool_correspond_ft_tf(u, v, w=w)
-    return float((ntf + nft) / np.array(2.0 * ntt + ntf + nft))
+    denom = 2.0 * ntt + ntf + nft
+    return float((ntf + nft) / denom) if denom != 0 else 0.0


 def rogerstanimoto(u, v, w=None):
```