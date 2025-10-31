# Bug Report: IntegerArray Power Operation with Base=1 and Negative Exponent

**Target**: `pandas.arrays.IntegerArray.__pow__`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

IntegerArray raises ValueError when computing `1 ** (negative integer)`, even though this operation is mathematically well-defined and always equals 1. The code has special-case handling for this scenario but applies it after the computation, which fails first.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
import pandas as pd


@given(st.lists(st.integers(min_value=-10, max_value=10) | st.none(), min_size=1, max_size=30))
@settings(max_examples=500)
def test_integerarray_one_pow_x_is_one(exponents):
    arr = pd.array(exponents, dtype="Int64")
    base = pd.array([1] * len(arr), dtype="Int64")
    result = base ** arr

    for i in range(len(result)):
        if pd.notna(result[i]):
            assert result[i] == 1
```

**Failing input**: `exponents=[-1]`

## Reproducing the Bug

```python
import pandas as pd

base = pd.array([1], dtype="Int64")
exponent = pd.array([-1], dtype="Int64")
result = base ** exponent
```

**Output:**
```
ValueError: Integers to negative integer powers are not allowed.
```

## Why This Is A Bug

Mathematically, `1 ** x = 1` for any value of x (including negative integers). The operation should succeed and return 1.

Pandas already has special-case code in `pandas/core/arrays/masked.py:810-811` that recognizes when the base is 1 and optimizes the result. However, this optimization is applied AFTER the power operation is computed at line 807, causing numpy to raise ValueError before the special case can be applied.

This also affects the rpow case (`1 ** arr` when arr contains negative integers).

## Fix

```diff
--- a/pandas/core/arrays/masked.py
+++ b/pandas/core/arrays/masked.py
@@ -799,13 +799,31 @@ class BaseMaskedArray(OpsMixin, ExtensionArray):
             # Make sure we do this before the "pow" mask checks
             #  to get an expected exception message on shape mismatch.
             if self.dtype.kind in "iu" and op_name in ["floordiv", "mod"]:
                 # TODO(GH#30188) ATM we don't match the behavior of non-masked
                 #  types with respect to floordiv-by-zero
                 pd_op = op

+            # Handle special cases for integer power operations to avoid ValueError
+            if op_name in ["pow", "rpow"] and self.dtype.kind in "iu":
+                # For integer dtypes, 1**x=1 and x**0=1, but numpy raises error
+                # for negative exponents. Compute these cases separately.
+                result = np.empty_like(self._data)
+                if op_name == "pow":
+                    safe_mask = (self._data != 1) & (other != 0)
+                    result[~safe_mask] = 1
+                    if safe_mask.any():
+                        result[safe_mask] = pd_op(self._data[safe_mask], other[safe_mask] if not is_scalar(other) else other)
+                else:  # rpow
+                    safe_mask = (other != 1) & (self._data != 0)
+                    result[~safe_mask] = 1
+                    if safe_mask.any():
+                        result[safe_mask] = pd_op(self._data[safe_mask], other)
+            else:
             with np.errstate(all="ignore"):
                 result = pd_op(self._data, other)

         if op_name == "pow":
```