# Bug Report: pandas.api.types.is_re_compilable Raises Exception on Invalid Regex

**Target**: pandas.api.types.is_re_compilable
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The is_re_compilable function raises re.PatternError exceptions for invalid regex patterns instead of returning False, violating its documented contract to return a bool.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas.api.types as pt
import re


@given(st.text())
def test_is_re_compilable_consistency_with_re_compile(pattern):
    is_compilable = pt.is_re_compilable(pattern)
    try:
        re.compile(pattern)
        can_compile = True
    except re.error:
        can_compile = False
    assert is_compilable == can_compile
```

**Failing inputs**: "[", ")", "?", and other invalid regex patterns

## Reproducing the Bug

```python
import pandas.api.types as pt

print(pt.is_re_compilable("["))
```

Expected: False  
Actual: re.PatternError: unterminated character set at position 0

## Why This Is A Bug

The functions docstring explicitly states it returns a bool indicating "Whether obj can be compiled as a regex pattern." It should return False for invalid regex patterns, not raise exceptions. The function currently only catches TypeError but not re.error (which includes re.PatternError).

## Fix

```diff
--- a/pandas/core/dtypes/inference.py
+++ b/pandas/core/dtypes/inference.py
@@ -184,7 +184,7 @@ def is_re_compilable(obj) -> bool:
     try:
         re.compile(obj)
-    except TypeError:
+    except (TypeError, re.error):
         return False
     else:
         return True
```
