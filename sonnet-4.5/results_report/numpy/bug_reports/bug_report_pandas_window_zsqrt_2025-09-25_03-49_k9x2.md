# Bug Report: pandas.core.window.common.zsqrt Python Scalar Handling

**Target**: `pandas.core.window.common.zsqrt`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `zsqrt` function in `pandas.core.window.common` crashes when passed a Python float scalar, while it correctly handles numpy scalars and arrays.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st
from pandas.core.window.common import zsqrt


@given(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10))
def test_zsqrt_always_nonnegative(x):
    result = zsqrt(x)
    assert result >= 0
```

**Failing input**: `x=0.0` (or any Python float)

## Reproducing the Bug

```python
from pandas.core.window.common import zsqrt

result = zsqrt(5.0)
```

Output:
```
AttributeError: 'bool' object has no attribute 'any'
```

## Why This Is A Bug

The `zsqrt` function is designed to compute the square root while safely handling negative values by setting them to 0. The function correctly handles:
- pandas DataFrames
- pandas Series
- numpy arrays
- numpy scalar types (e.g., `np.float64`)

However, it fails on Python float scalars with an AttributeError. The bug occurs because when `x` is a Python float, the expression `mask = x < 0` produces a Python `bool` object, which doesn't have an `.any()` method that the code attempts to call at line 158.

This inconsistent handling of numeric types is problematic because:
1. Python floats are a legitimate numeric type that numpy operations often produce
2. The function doesn't document any restrictions on input types
3. Users would reasonably expect the function to work with all scalar numeric types
4. numpy operations like `np.sqrt(5.0)` can return Python floats in certain contexts

## Fix

```diff
--- a/pandas/core/window/common.py
+++ b/pandas/core/window/common.py
@@ -149,13 +149,18 @@ def flex_binary_moment(arg1, arg2, f, pairwise: bool = False):
 def zsqrt(x):
     with np.errstate(all="ignore"):
         result = np.sqrt(x)
         mask = x < 0

     if isinstance(x, ABCDataFrame):
         if mask._values.any():
             result[mask] = 0
     else:
-        if mask.any():
+        # Handle both array-like (with .any() method) and scalar booleans
+        if isinstance(mask, (bool, np.bool_)):
+            if mask:
+                result = 0
+        elif mask.any():
             result[mask] = 0

     return result
```