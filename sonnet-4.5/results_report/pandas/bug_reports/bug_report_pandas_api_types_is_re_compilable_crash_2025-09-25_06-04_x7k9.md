# Bug Report: pandas.api.types.is_re_compilable Crashes on Invalid Regex

**Target**: `pandas.api.types.is_re_compilable`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `is_re_compilable` function crashes with `re.PatternError` when given invalid regex patterns, instead of returning `False` as documented. The function only catches `TypeError` but not `re.error` exceptions.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pandas.api import types as pat
import re

@given(st.text(min_size=1))
@settings(max_examples=100)
def test_is_re_compilable_returns_bool(pattern):
    try:
        re.compile(pattern)
        expected = True
    except re.error:
        expected = False

    result = pat.is_re_compilable(pattern)
    assert isinstance(result, bool), f"is_re_compilable should return bool, not raise exception"
    assert result == expected, f"is_re_compilable mismatch for pattern: {pattern!r}"
```

**Failing input**: `')'` (and other invalid patterns like `'?'`, `'*'`, `'('`)

## Reproducing the Bug

```python
from pandas.api import types as pat

print(pat.is_re_compilable(')'))
```

**Output**:
```
re.PatternError: unbalanced parenthesis at position 0
```

**Expected output**: `False`

## Why This Is A Bug

The function's docstring explicitly states it returns `bool`:

```
Returns
-------
bool
    Whether `obj` can be compiled as a regex pattern.
```

The examples show it should return `False` for non-compilable inputs:
```python
>>> is_re_compilable(".*")
True
>>> is_re_compilable(1)
False
```

Invalid regex patterns are non-compilable inputs and should return `False`, not raise an exception. Users calling this function expect a boolean return value to check if a pattern is valid before using it.

## Fix

```diff
--- a/pandas/core/dtypes/inference.py
+++ b/pandas/core/dtypes/inference.py
@@ -186,7 +186,7 @@ def is_re_compilable(obj) -> bool:
     """
     try:
         re.compile(obj)
-    except TypeError:
+    except (TypeError, re.error):
         return False
     else:
         return True
```

This change makes the function catch both `TypeError` (for non-string inputs like `1`) and `re.error` (for invalid regex patterns like `')'`), ensuring it always returns a boolean as documented.