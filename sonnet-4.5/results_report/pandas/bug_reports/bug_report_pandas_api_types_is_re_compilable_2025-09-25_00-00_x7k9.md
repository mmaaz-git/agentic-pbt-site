# Bug Report: pandas.api.types.is_re_compilable Crashes on Invalid Regex Patterns

**Target**: `pandas.api.types.is_re_compilable`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `is_re_compilable` function is documented to "Check if the object can be compiled into a regex pattern instance" and return a boolean. However, it crashes with `re.PatternError` when given invalid regex patterns instead of returning `False`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.api import types
import re


@given(st.text(min_size=1, max_size=100))
def test_is_re_compilable_should_not_raise(pattern_str):
    result = types.is_re_compilable(pattern_str)

    assert isinstance(result, bool)

    if result:
        re.compile(pattern_str)
```

**Failing input**: `'?'` (and many other invalid regex patterns like `'*'`, `'+'`, `'('`, `'['`, etc.)

## Reproducing the Bug

```python
from pandas.api.types import is_re_compilable

invalid_patterns = ['?', '*', '+', '\\', '[', '(', ')']

for pattern in invalid_patterns:
    try:
        result = is_re_compilable(pattern)
        print(f"{pattern}: {result}")
    except Exception as e:
        print(f"{pattern}: CRASH - {type(e).__name__}")
```

## Why This Is A Bug

The function's docstring states it should "Check if the object can be compiled into a regex pattern instance" and return a `bool`. The function signature is `is_re_compilable(obj) -> bool`, indicating it should always return a boolean value.

However, when given invalid regex patterns, it raises `re.PatternError` instead of returning `False`. This violates the documented contract and makes the function unreliable for its intended purpose of checking compilability.

## Fix

The function currently only catches `TypeError`, but should also catch `re.error` (which `re.PatternError` inherits from):

```diff
--- a/pandas/core/dtypes/inference.py
+++ b/pandas/core/dtypes/inference.py
@@ -185,7 +185,7 @@ def is_re_compilable(obj) -> bool:
         False
     """
     try:
         re.compile(obj)
-    except TypeError:
+    except (TypeError, re.error):
         return False
     else:
         return True
```