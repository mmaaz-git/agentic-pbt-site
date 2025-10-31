# Bug Report: scipy.spatial.distance.dice Identity Property Violation with All-False Arrays

**Target**: `scipy.spatial.distance.dice`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `dice` dissimilarity function returns `nan` instead of `0.0` when both input boolean arrays are all False, violating the fundamental identity property that `d(x, x) = 0` for all dissimilarity measures.

## Property-Based Test

```python
import numpy as np
import scipy.spatial.distance as dist
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays


@given(arrays(np.bool_, (10,), elements=st.booleans()))
def test_dice_identity_property(u):
    d = dist.dice(u, u)
    assert not np.isnan(d), f"dice(u, u) should not be NaN, got {d} for u={u}"


if __name__ == "__main__":
    test_dice_identity_property()
```

<details>

<summary>
**Failing input**: `array([False, False, False, False, False, False, False, False, False, False])`
</summary>
```
/home/npc/.local/lib/python3.13/site-packages/scipy/spatial/distance.py:1503: RuntimeWarning: invalid value encountered in divide
  return float((ntf + nft) / np.array(2.0 * ntt + ntf + nft))
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/55/hypo.py", line 14, in <module>
    test_dice_identity_property()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/55/hypo.py", line 8, in test_dice_identity_property
    def test_dice_identity_property(u):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/55/hypo.py", line 10, in test_dice_identity_property
    assert not np.isnan(d), f"dice(u, u) should not be NaN, got {d} for u={u}"
           ^^^^^^^^^^^^^^^
AssertionError: dice(u, u) should not be NaN, got nan for u=[False False False False False False False False False False]
Falsifying example: test_dice_identity_property(
    u=array([False, False, False, False, False, False, False, False, False,
           False]),
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/arrayprint.py:1708
```
</details>

## Reproducing the Bug

```python
import numpy as np
import scipy.spatial.distance as dist

u = np.array([False, False, False, False, False])
v = np.array([False, False, False, False, False])

result = dist.dice(u, v)

assert np.array_equal(u, v), f"Arrays are not equal: u={u}, v={v}"
print(f"dice(u, u) = {result}")
print(f"Is result NaN? {np.isnan(result)}")
print(f"Arrays are identical: {np.array_equal(u, v)}")
```

<details>

<summary>
Output shows NaN despite identical arrays
</summary>
```
/home/npc/.local/lib/python3.13/site-packages/scipy/spatial/distance.py:1503: RuntimeWarning: invalid value encountered in divide
  return float((ntf + nft) / np.array(2.0 * ntt + ntf + nft))
dice(u, u) = nan
Is result NaN? True
Arrays are identical: True
```
</details>

## Why This Is A Bug

The Dice dissimilarity is mathematically defined as:

```
dice(u, v) = (c_TF + c_FT) / (2*c_TT + c_TF + c_FT)
```

where `c_ij` represents the count of positions where `u[k] = i` and `v[k] = j`.

When both arrays are all False:
- `c_TT = 0` (no positions where both are True)
- `c_TF = 0` (no positions where u=True, v=False)
- `c_FT = 0` (no positions where u=False, v=True)

This results in `0/0`, which produces NaN due to division by zero.

However, this violates a fundamental axiom of dissimilarity measures: **the identity property** states that `d(x, x) = 0` for any vector x. Even though the arrays represent "empty sets" with no True values, they are identical vectors, so their dissimilarity must be 0.0, not NaN.

The documentation does not explicitly address this edge case, but mathematical principles and consistency with related functions demand that identical vectors have zero dissimilarity.

## Relevant Context

1. **Mathematical Axiom Violation**: All dissimilarity/distance measures must satisfy `d(x, x) = 0` (the identity axiom). This is a fundamental requirement in metric spaces and dissimilarity theory.

2. **Inconsistency with Related Functions**: The Jaccard dissimilarity in the same module (`scipy.spatial.distance.jaccard`) correctly handles this case. See the implementation at `/home/npc/pbt/agentic-pbt/envs/scipy_env/lib/python3.13/site-packages/scipy/spatial/distance.py:915`:
   ```python
   return (a / b) if b != 0 else np.float64(0)
   ```
   The Jaccard documentation even explicitly states: "If u and v are both zero, their Jaccard dissimilarity is defined to be zero." (line 815-816)

3. **Mathematical Relationship**: The Dice and Jaccard coefficients are mathematically related: `Dice = 2*Jaccard/(1+Jaccard)`. It's inconsistent for these related measures to handle the same edge case differently.

4. **Practical Impact**: This bug can cause unexpected failures in production code that:
   - Compares boolean feature vectors that may be empty
   - Uses dice dissimilarity in clustering algorithms
   - Performs similarity searches where some vectors have no active features

5. **Warning Generated**: The current implementation generates a RuntimeWarning about invalid division, which is a code smell indicating unhandled edge cases.

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
+    return float((ntf + nft) / denominator)
```