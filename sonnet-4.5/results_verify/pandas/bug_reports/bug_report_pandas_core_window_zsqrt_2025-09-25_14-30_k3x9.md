# Bug Report: pandas.core.window.common.zsqrt Crashes on Scalar Inputs

**Target**: `pandas.core.window.common.zsqrt`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `zsqrt` function in pandas.core.window.common crashes with an AttributeError when called with scalar numeric inputs instead of Series/DataFrame objects.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.core.window.common import zsqrt

@given(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10))
def test_zsqrt_non_negative_scalar(x):
    result = zsqrt(x)
    assert result >= 0
```

**Failing input**: `x=0.0` (and any other scalar float value)

## Reproducing the Bug

```python
from pandas.core.window.common import zsqrt

zsqrt(0.0)
```

## Why This Is A Bug

The `zsqrt` function attempts to compute a "zero-safe" square root (returning 0 for negative inputs). However, it assumes inputs are always Series, DataFrame, or numpy arrays. When called with a scalar:

1. `mask = x < 0` produces a plain boolean (e.g., `False`)
2. The code then calls `mask.any()`, but booleans don't have an `.any()` method
3. This results in `AttributeError: 'bool' object has no attribute 'any'`

The function lacks:
- Input type validation
- Type hints documenting expected inputs
- Support for scalar inputs (which would be logical given the function's purpose)

While current callers may only pass Series/DataFrame/arrays, the function's interface doesn't document this restriction, making it a footgun for future use.

## Fix

```diff
--- a/pandas/core/window/common.py
+++ b/pandas/core/window/common.py
@@ -149,12 +149,17 @@ def flex_binary_moment(arg1, arg2, f, pairwise: bool = False):
 def zsqrt(x):
     with np.errstate(all="ignore"):
         result = np.sqrt(x)
         mask = x < 0

     if isinstance(x, ABCDataFrame):
         if mask._values.any():
             result[mask] = 0
     else:
-        if mask.any():
+        # Handle both array-like and scalar inputs
+        if np.ndim(mask) == 0:
+            # Scalar case
+            if mask:
+                result = 0
+        elif mask.any():
             result[mask] = 0

     return result
```