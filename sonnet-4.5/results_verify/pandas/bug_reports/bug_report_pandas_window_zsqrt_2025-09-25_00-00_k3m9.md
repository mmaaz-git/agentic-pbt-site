# Bug Report: pandas.core.window.common.zsqrt Crashes on Scalar Input

**Target**: `pandas.core.window.common.zsqrt`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `zsqrt` function crashes with `AttributeError` when given scalar numeric inputs instead of pandas Series or DataFrame objects.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.core.window.common import zsqrt


@given(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10))
def test_zsqrt_always_nonnegative(x):
    result = zsqrt(x)
    assert result >= 0, f"zsqrt({x}) = {result} is negative"
```

**Failing input**: `x=0.0` (or any scalar value)

## Reproducing the Bug

```python
from pandas.core.window.common import zsqrt

zsqrt(0.0)
```

Output:
```
AttributeError: 'bool' object has no attribute 'any'
```

## Why This Is A Bug

The `zsqrt` function is designed to compute square root with special handling for negative values (returning 0 instead of NaN). While it works correctly with pandas Series and DataFrame objects, it fails on scalar numeric inputs.

The issue is on line 158 of `pandas/core/window/common.py`:
- When `x` is a scalar, `mask = x < 0` produces a Python `bool` (not a numpy array)
- Calling `mask.any()` on a `bool` raises `AttributeError`

This is a legitimate bug because:
1. The function has no type hints or documentation restricting inputs to pandas objects
2. It's used in contexts where scalar results are possible (e.g., `zsqrt(x_var * y_var)` in ewm.py:892)
3. The conceptual purpose (safe square root) naturally applies to scalars

## Fix

```diff
--- a/pandas/core/window/common.py
+++ b/pandas/core/window/common.py
@@ -149,12 +149,16 @@ def zsqrt(x):
 def zsqrt(x):
     with np.errstate(all="ignore"):
         result = np.sqrt(x)
         mask = x < 0

     if isinstance(x, ABCDataFrame):
         if mask._values.any():
             result[mask] = 0
     else:
-        if mask.any():
+        # Handle both scalar and array-like inputs
+        if np.ndim(mask) == 0:  # scalar
+            if mask:
+                result = 0
+        elif mask.any():  # array-like
             result[mask] = 0

     return result
```