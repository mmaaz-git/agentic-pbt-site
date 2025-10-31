# Bug Report: pandas.api.types.is_re_compilable Crashes on Invalid Regex

**Target**: `pandas.api.types.is_re_compilable`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`is_re_compilable()` is documented to return a boolean indicating whether an object can be compiled as a regex, but it crashes with `re.PatternError` when given invalid regex patterns instead of returning `False`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.api import types
import re

@given(st.one_of(st.text(), st.binary()))
def test_is_re_compilable_matches_re_compile(obj):
    result = types.is_re_compilable(obj)
    try:
        re.compile(obj)
        actually_compilable = True
    except (TypeError, re.error):
        actually_compilable = False

    assert result == actually_compilable
```

**Failing inputs**: `'['`, `')'`, `'?'`, and many other invalid regex patterns

## Reproducing the Bug

```python
from pandas.api.types import is_re_compilable

print(is_re_compilable('['))
```

Expected output: `False`
Actual output: `re.PatternError: unterminated character set at position 0`

## Why This Is A Bug

The function's docstring states it should return a boolean value indicating "Whether `obj` can be compiled as a regex pattern." The function signature is `is_re_compilable(obj) -> bool`. However, when given invalid regex patterns, the function crashes with `re.PatternError` instead of returning `False`.

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