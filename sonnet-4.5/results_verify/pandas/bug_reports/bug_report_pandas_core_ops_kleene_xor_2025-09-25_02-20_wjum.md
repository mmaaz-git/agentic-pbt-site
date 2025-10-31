# Bug Report: pandas.core.ops.kleene_xor NA ^ NA Returns NA Instead of False

**Target**: `pandas.core.ops.kleene_xor`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `kleene_xor` function incorrectly returns NA for `NA ^ NA` when it should return False, violating the fundamental XOR property that `x ^ x = False` for all x.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings

@given(
    st.lists(st.booleans(), min_size=1, max_size=20),
    st.lists(st.booleans(), min_size=1, max_size=20),
)
@settings(max_examples=500)
def test_kleene_xor_self_is_false(values, mask_vals):
    n = max(len(values), len(mask_vals))
    left = np.array(values[:n] + [False] * (n - len(values)))
    left_mask = np.array(mask_vals[:n] + [False] * (n - len(mask_vals)))

    result, mask = kleene_xor(left, left, left_mask, left_mask)

    assert np.all(~result), f"x ^ x should be False everywhere, got {result}"
    assert np.all(~mask), f"x ^ x should not be masked (even for NA), got mask={mask}"
```

**Failing input**: `values=[False], mask_vals=[True]`

## Reproducing the Bug

```python
import numpy as np
from pandas.core.ops import kleene_xor

left = np.array([False])
left_mask = np.array([True])

result, mask = kleene_xor(left, left, left_mask, left_mask)

print(f"NA ^ NA returns: result={result[0]}, mask={mask[0]}")
print(f"Expected: result=False, mask=False")
assert not mask[0], "NA ^ NA should return False (unmasked), not NA"
```

The assertion fails, showing that `NA ^ NA` returns `mask=True` (NA) instead of `mask=False` (False).

High-level demonstration:

```python
import pandas as pd

s = pd.Series([True, False, pd.NA], dtype="boolean")
result = s ^ s

print(result.to_list())
```

Output: `[False, False, <NA>]`
Expected: `[False, False, False]`

## Why This Is A Bug

The XOR operation has the fundamental property that `x ^ x = False` for any value x. This holds regardless of whether x is known or unknown:

- If x = True: `True ^ True = False`
- If x = False: `False ^ False = False`
- If x = NA (unknown): `x ^ x = False` because x is the **same** unknown value

The current implementation treats `NA ^ NA` as two independent unknowns (like `NA₁ ^ NA₂`), returning NA. However, when computing `kleene_xor(left, left, left_mask, left_mask)`, the NAs at position i in both arguments represent the **same** unknown value, so the result should deterministically be False.

This violates the idempotent-like property of XOR where `x ^ x` always equals the identity element (False).

## Fix

The bug is in these lines of `kleene_xor`:

```python
else:
    mask = left_mask | right_mask
```

This produces NA (masked=True) whenever either input is NA. For XOR, when both positions are masked AND the values are from the same source, the result should be False (unmasked).

A potential fix:

```diff
diff --git a/pandas/core/ops/mask_ops.py b/pandas/core/ops/mask_ops.py
index xxxx..yyyy 100644
--- a/pandas/core/ops/mask_ops.py
+++ b/pandas/core/ops/mask_ops.py
@@ -xxx,x +xxx,x @@ def kleene_xor(
     if right_mask is None:
         if right is libmissing.NA:
             mask = np.ones_like(left_mask)
         else:
             mask = left_mask.copy()
     else:
-        mask = left_mask | right_mask
+        # XOR has the property that x ^ x = False even when x is unknown
+        # When both inputs are masked, check if values are equal
+        both_masked = left_mask & right_mask
+        values_equal = left == right
+        # Result is unmasked (False) where both are masked but values are equal
+        mask = (left_mask | right_mask) & ~(both_masked & values_equal)

     return result, mask
```

Note: This fix assumes that when `left` and `right` have equal values at a position and both are masked, they represent the same unknown value. This handles the `x ^ x` case correctly. For truly independent unknowns with different values, the behavior remains unchanged.