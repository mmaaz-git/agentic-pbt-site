# Bug Report: pandas.api.types.is_re_compilable Crashes on Invalid Regex Patterns

**Target**: `pandas.api.types.is_re_compilable`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `is_re_compilable` function crashes with `re.PatternError` when given invalid regex patterns instead of returning `False` as documented.

## Property-Based Test

```python
import pandas.api.types as types
import re
from hypothesis import given, strategies as st


@given(st.text(min_size=1))
def test_is_re_compilable_for_valid_patterns(pattern):
    try:
        re.compile(pattern)
        can_compile = True
    except re.error:
        can_compile = False

    result = types.is_re_compilable(pattern)
    assert result == can_compile
```

**Failing input**: `'('`

## Reproducing the Bug

```python
import pandas.api.types as types

print(types.is_re_compilable('('))
```

This crashes with:
```
re.PatternError: missing ), unterminated subpattern at position 0
```

Other failing inputs: `')'`, `'?'`, `'*'`, `'['`

## Why This Is A Bug

The function's docstring states it should "Check if the object can be compiled into a regex pattern instance" and return a bool. The documented behavior is to return `True` if the pattern can be compiled, and `False` otherwise (as shown in the example with `is_re_compilable(1)` returning `False`).

However, the function crashes on invalid regex patterns instead of returning `False`. This violates the function's contract of always returning a bool.

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