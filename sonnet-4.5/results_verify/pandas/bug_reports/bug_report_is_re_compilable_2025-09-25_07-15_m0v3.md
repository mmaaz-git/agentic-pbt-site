# Bug Report: is_re_compilable Crashes on Invalid Regex Patterns

**Target**: `pandas.api.types.is_re_compilable`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `is_re_compilable` function is supposed to check if an object can be compiled into a regex pattern and return a boolean. However, it crashes with `re.PatternError` when given invalid regex patterns instead of returning `False`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas.api.types as pt
import re

@given(st.text())
@settings(max_examples=300)
def test_is_re_compilable_on_strings(s):
    result = pt.is_re_compilable(s)

    try:
        re.compile(s)
        can_compile = True
    except re.error:
        can_compile = False

    assert result == can_compile
```

**Failing input**: `'?'`

## Reproducing the Bug

```python
import pandas.api.types as pt

result = pt.is_re_compilable('?')
```

This crashes with:
```
re.PatternError: nothing to repeat at position 0
```

Expected behavior: Should return `False` instead of crashing.

Other examples that crash:
```python
pt.is_re_compilable('*')
pt.is_re_compilable('+')
pt.is_re_compilable('(?')
pt.is_re_compilable('[')
```

All of these raise `re.PatternError` instead of returning `False`.

## Why This Is A Bug

The function's purpose (from its docstring) is to "Check if the object can be compiled into a regex pattern instance" and return a boolean. Users expect a boolean return value, not an exception for invalid regex patterns.

The current implementation only catches `TypeError` but not `re.error`:

```python
def is_re_compilable(obj) -> bool:
    try:
        re.compile(obj)
    except TypeError:
        return False
    else:
        return True
```

## Fix

The fix should catch `re.error` (the base class for all regex compilation errors) in addition to `TypeError`:

```diff
--- a/pandas/core/dtypes/inference.py
+++ b/pandas/core/dtypes/inference.py
@@ -185,7 +185,7 @@ def is_re_compilable(obj) -> bool:
     """
     try:
         re.compile(obj)
-    except TypeError:
+    except (TypeError, re.error):
         return False
     else:
         return True
```
