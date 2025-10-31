# Bug Report: pandas.core.ops Kleene Logic Functions Infinite Recursion

**Target**: `pandas.core.ops.kleene_and`, `pandas.core.ops.kleene_or`, `pandas.core.ops.kleene_xor`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The Kleene logic functions (`kleene_and`, `kleene_or`, `kleene_xor`) enter infinite recursion when both `left_mask` and `right_mask` parameters are `None`, causing a `RecursionError`. While the docstring states "Only one of these may be None", the functions do not validate this precondition and instead crash with infinite recursion.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings
from pandas.core.ops import kleene_and

@given(st.lists(st.booleans(), min_size=1, max_size=50),
       st.lists(st.booleans(), min_size=1, max_size=50))
@settings(max_examples=500)
def test_kleene_and_without_na(left_vals, right_vals):
    min_len = min(len(left_vals), len(right_vals))
    left = np.array(left_vals[:min_len])
    right = np.array(right_vals[:min_len])
    result, mask = kleene_and(left, right, None, None)
    expected = left & right
    assert np.array_equal(result, expected)
```

**Failing input**: `left=[True, False]`, `right=[True, True]`, `left_mask=None`, `right_mask=None`

## Reproducing the Bug

```python
import numpy as np
from pandas.core.ops import kleene_and, kleene_or, kleene_xor

left = np.array([True, False])
right = np.array([True, True])

kleene_and(left, right, None, None)

kleene_or(left, right, None, None)

kleene_xor(left, right, None, None)
```

## Why This Is A Bug

The functions document that "Only one of these may be None" but do not enforce this precondition. Instead, when both masks are `None`, the code enters infinite recursion:

1. `kleene_and(left, right, None, None)` checks `if left_mask is None`
2. It recursively calls `kleene_and(right, left, right_mask, left_mask)` which is `kleene_and(right, left, None, None)`
3. The new call again finds `left_mask is None` and swaps arguments back
4. This continues infinitely until `RecursionError`

The same logic flaw exists in all three functions (`kleene_and`, `kleene_or`, `kleene_xor`). Functions should either:
- Properly handle the case where both masks are `None`
- Validate the precondition and raise a clear `ValueError` immediately

## Fix

```diff
--- a/pandas/core/ops/mask_ops.py
+++ b/pandas/core/ops/mask_ops.py
@@ -138,6 +138,11 @@ def kleene_and(
     The result of the logical xor, and the new mask.
     """
+    # Validate precondition
+    if left_mask is None and right_mask is None:
+        # No NA values, compute standard boolean and
+        result = left & right
+        return result, np.zeros(len(result) if hasattr(result, '__len__') else 1, dtype=bool)
+
     # To reduce the number of cases, we ensure that `left` & `left_mask`
     # always come from an array, not a scalar. This is safe, since
     # A & B == B & A
@@ -188,6 +193,11 @@ def kleene_or(
     The result of the logical or, and the new mask.
     """
+    # Validate precondition
+    if left_mask is None and right_mask is None:
+        # No NA values, compute standard boolean or
+        result = left | right
+        return result, np.zeros(len(result) if hasattr(result, '__len__') else 1, dtype=bool)
+
     if left_mask is None:
         return kleene_or(right, left, right_mask, left_mask)

@@ -75,6 +75,11 @@ def kleene_xor(
     The result of the logical xor, and the new mask.
     """
+    # Validate precondition
+    if left_mask is None and right_mask is None:
+        # No NA values, compute standard boolean xor
+        result = left ^ right
+        return result, np.zeros(len(result) if hasattr(result, '__len__') else 1, dtype=bool)
+
     if left_mask is None:
         return kleene_xor(right, left, right_mask, left_mask)