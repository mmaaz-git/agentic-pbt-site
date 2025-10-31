# Bug Report: pandas.core.window.common.zsqrt Crashes on Scalar Input

**Target**: `pandas.core.window.common.zsqrt`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `zsqrt` function in `pandas.core.window.common` crashes with `AttributeError` when passed scalar numeric values, despite using `np.sqrt` which accepts scalars and having no documented restriction against scalar inputs.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.core.window.common import zsqrt

@given(st.floats(allow_nan=False, allow_infinity=False, min_value=0, max_value=1e10))
def test_zsqrt_nonnegative_roundtrip(x):
    result = zsqrt(x)
    assert result >= 0
```

**Failing input**: `x=0.0` (or any scalar value)

## Reproducing the Bug

```python
from pandas.core.window.common import zsqrt

result = zsqrt(4.0)
```

**Output:**
```
AttributeError: 'bool' object has no attribute 'any'
```

**Traceback:**
```
File "pandas/core/window/common.py", line 158, in zsqrt
    if mask.any():
       ^^^^^^^^
AttributeError: 'bool' object has no attribute 'any'
```

## Why This Is A Bug

The `zsqrt` function is designed to compute `sqrt(x)` while handling negative values by returning 0. The implementation:

1. Uses `np.sqrt(x)` which accepts scalars
2. Has no documentation restricting inputs to Series/DataFrame only
3. Crashes on scalar inputs because `mask = x < 0` produces a Python `bool` for scalars, which lacks the `.any()` method

When `x` is a scalar, `mask = x < 0` evaluates to a boolean value (e.g., `True` or `False`), not an array. The code then tries to call `mask.any()` on line 158, but Python booleans don't have this method.

The function correctly handles DataFrames (lines 154-156) and accidentally works for Series (which do have `.any()`), but fails for scalar inputs.

## Fix

```diff
--- a/pandas/core/window/common.py
+++ b/pandas/core/window/common.py
@@ -149,13 +149,16 @@ def flex_binary_moment(arg1, arg2, f, pairwise: bool = False):
 def zsqrt(x):
     with np.errstate(all="ignore"):
         result = np.sqrt(x)
         mask = x < 0

     if isinstance(x, ABCDataFrame):
         if mask._values.any():
             result[mask] = 0
-    else:
+    elif hasattr(mask, 'any'):
         if mask.any():
             result[mask] = 0
+    else:
+        if mask:
+            result = 0

     return result
```