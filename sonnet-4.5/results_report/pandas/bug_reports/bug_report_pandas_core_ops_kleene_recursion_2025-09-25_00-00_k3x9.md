# Bug Report: pandas.core.ops Kleene Logic Functions Infinite Recursion

**Target**: `pandas.core.ops.kleene_and`, `pandas.core.ops.kleene_or`, `pandas.core.ops.kleene_xor`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

All three Kleene logic functions (`kleene_and`, `kleene_or`, `kleene_xor`) crash with `RecursionError` when both `left_mask` and `right_mask` are `None`, instead of validating the documented precondition.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings
from pandas.core.ops import kleene_and


@st.composite
def bool_arrays_with_masks(draw):
    size = draw(st.integers(min_value=1, max_value=100))
    values = draw(st.lists(st.booleans(), min_size=size, max_size=size))
    mask_presence = draw(st.booleans())
    if mask_presence:
        mask_values = draw(st.lists(st.booleans(), min_size=size, max_size=size))
        mask = np.array(mask_values, dtype=bool)
    else:
        mask = None
    return np.array(values, dtype=bool), mask


@settings(max_examples=500)
@given(bool_arrays_with_masks(), bool_arrays_with_masks())
def test_kleene_and_commutativity_arrays(left_data, right_data):
    left, left_mask = left_data
    right, right_mask = right_data

    if len(left) != len(right):
        return

    result1, mask1 = kleene_and(left, right, left_mask, right_mask)
    result2, mask2 = kleene_and(right, left, right_mask, left_mask)

    assert np.array_equal(result1, result2)
    assert np.array_equal(mask1, mask2)
```

**Failing input**: Arrays with both `left_mask=None` and `right_mask=None`

## Reproducing the Bug

```python
import numpy as np
from pandas.core.ops import kleene_and

arr1 = np.array([True, False], dtype=bool)
arr2 = np.array([True, True], dtype=bool)

kleene_and(arr1, arr2, None, None)
```

## Why This Is A Bug

The docstring states: "Only one of these may be None, which implies that the associated `left` or `right` value is a scalar." However, when both masks are `None`, the code enters infinite recursion:

1. Line 157: `if left_mask is None: return kleene_and(right, left, right_mask, left_mask)`
2. This swaps arguments, but `right_mask` is also `None`, so the recursive call has `left_mask=None` again
3. Infinite recursion continues until stack overflow

The same bug exists in `kleene_or` and `kleene_xor` at the same logical location.

Instead of a confusing `RecursionError`, the functions should validate the precondition and raise a clear error message like: `ValueError: At least one of left_mask or right_mask must be provided`.

## Fix

```diff
--- a/pandas/core/ops/mask_ops.py
+++ b/pandas/core/ops/mask_ops.py
@@ -154,6 +154,9 @@ def kleene_and(
     # To reduce the number of cases, we ensure that `left` & `left_mask`
     # always come from an array, not a scalar. This is safe, since
     # A & B == B & A
+    if left_mask is None and right_mask is None:
+        raise ValueError(
+            "At least one of left_mask or right_mask must be provided"
+        )
     if left_mask is None:
         return kleene_and(right, left, right_mask, left_mask)
```

Apply the same fix to `kleene_or` and `kleene_xor`.