# Bug Report: zsqrt fails on scalar inputs

**Target**: `pandas.core.window.common.zsqrt`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `zsqrt` function crashes with `AttributeError` when called with scalar float values instead of Series/DataFrame objects.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.core.window.common import zsqrt

@given(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False))
def test_zsqrt_always_nonnegative(x):
    result = zsqrt(x)
    assert result >= 0
```

**Failing input**: `x=0.0` (or any scalar float value)

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

The `zsqrt` function is defined in a public module (`pandas.core.window.common`) and has no documentation restricting its use to only Series/DataFrame inputs. The function works correctly with pandas Series and DataFrame objects, but crashes with an unclear error when passed scalar values.

The bug occurs at line 158 in common.py:
```python
def zsqrt(x):
    with np.errstate(all="ignore"):
        result = np.sqrt(x)
        mask = x < 0

    if isinstance(x, ABCDataFrame):
        if mask._values.any():
            result[mask] = 0
    else:
        if mask.any():  # Bug: when x is scalar, mask is bool, not array
            result[mask] = 0

    return result
```

When `x` is a scalar:
- `mask = x < 0` creates a boolean value (True or False)
- `mask.any()` fails because bool objects don't have an `any()` method
- The function should either handle scalars correctly or raise a clear error message

## Fix

```diff
--- a/pandas/core/window/common.py
+++ b/pandas/core/window/common.py
@@ -149,10 +149,17 @@ def flex_binary_moment(arg1, arg2, f, pairwise: bool = False):
 def zsqrt(x):
+    import numbers
     with np.errstate(all="ignore"):
         result = np.sqrt(x)
         mask = x < 0

     if isinstance(x, ABCDataFrame):
         if mask._values.any():
             result[mask] = 0
+    elif isinstance(x, ABCSeries):
+        if mask.any():
+            result[mask] = 0
+    elif isinstance(x, (numbers.Number, np.number)):
+        if mask:
+            result = 0
     else:
-        if mask.any():
-            result[mask] = 0
+        if mask.any():
+            result[mask] = 0

     return result
```