# Bug Report: pandas.api.types.is_re_compilable Crashes on Invalid Regex

**Target**: `pandas.api.types.is_re_compilable`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`is_re_compilable` raises `re.PatternError` for invalid regex patterns instead of returning `False` as its contract implies.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas.api.types as pat


@given(st.text())
@settings(max_examples=500)
def test_is_re_compilable_consistency(pattern):
    is_compilable = pat.is_re_compilable(pattern)

    if is_compilable:
        import re
        try:
            re.compile(pattern)
        except:
            assert False, f"is_re_compilable returned True but re.compile failed"
```

**Failing input**: `')'` (also `'?'`, `'*'`, `'('`, `'['`)

## Reproducing the Bug

```python
import pandas.api.types as pat

pat.is_re_compilable(')')
```

## Why This Is A Bug

The function's name and docstring clearly indicate it should **check** if an object can be compiled as a regex pattern and return a boolean. Instead, it crashes on invalid patterns like `')'`. The docstring states:

> Check if the object can be compiled into a regex pattern instance.
>
> Returns: bool - Whether `obj` can be compiled as a regex pattern.

A function that checks compilability should never crash on non-compilable input.

## Fix

```diff
--- a/pandas/core/dtypes/inference.py
+++ b/pandas/core/dtypes/inference.py
@@ -185,7 +185,7 @@ def is_re_compilable(obj) -> bool:
         False
     """
     try:
         re.compile(obj)
-    except TypeError:
+    except (TypeError, re.PatternError):
         return False
     else:
         return True
```