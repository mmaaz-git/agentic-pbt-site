# Bug Report: pandas.api.types.is_re_compilable Raises Exception Instead of Returning False

**Target**: `pandas.api.types.is_re_compilable`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `is_re_compilable` function is documented to return a bool indicating whether an object can be compiled as a regex pattern. However, it raises `re.PatternError` for invalid regex patterns instead of returning False.

## Property-Based Test

```python
import re
from hypothesis import given, strategies as st
import pandas.api.types as pat


@given(st.text())
def test_is_re_compilable_for_strings(x):
    result = pat.is_re_compilable(x)
    if result:
        try:
            re.compile(x)
        except re.error:
            raise AssertionError(f"is_re_compilable({x!r}) returned True but re.compile raised error")
```

**Failing input**: `'['` (and many other invalid regex patterns like `'?'`, `'(unclosed'`)

## Reproducing the Bug

```python
import pandas.api.types as pat

result = pat.is_re_compilable('[')
```

This raises:
```
re.PatternError: unterminated character set at position 0
```

But according to the function's docstring and signature, it should return `False`.

## Why This Is A Bug

The function's signature and docstring promise:
- **Returns**: `bool` - Whether `obj` can be compiled as a regex pattern
- **Examples show**: Returns `True` for valid patterns, `False` for invalid types (e.g., integers)

The function currently only catches `TypeError`, but `re.compile()` can also raise `re.PatternError` (a subclass of `re.error`) for syntactically invalid regex patterns. These should return `False`, not propagate as exceptions.

## Fix

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