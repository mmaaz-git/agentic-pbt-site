# Bug Report: pandas.core.dtypes.inference.is_re_compilable raises exception on invalid regex patterns

**Target**: `pandas.core.dtypes.inference.is_re_compilable`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `is_re_compilable` function raises `PatternError` exceptions when given invalid regex patterns, instead of returning `False` as documented. The function's docstring states it "returns bool" indicating whether an object can be compiled as a regex pattern, but it crashes on invalid patterns like `'('`, `')'`, `'['`, etc.

## Property-Based Test

```python
from pandas.core.dtypes.inference import is_re_compilable
from hypothesis import given, strategies as st
import re
import pytest


@given(st.text(min_size=1, max_size=50, alphabet='abcdefghijklmnopqrstuvwxyz0123456789.*+?[]()'))
def test_is_re_compilable_never_crashes(pattern):
    try:
        re.compile(pattern)
        expected = True
    except re.error:
        expected = False

    result = is_re_compilable(pattern)
    assert result == expected, f"is_re_compilable should match re.compile for pattern '{pattern}'"
```

**Failing input**: `'('` (and other invalid regex patterns like `')'`, `'['`, `'*'`, `'+'`, `'?'`)

## Reproducing the Bug

```python
from pandas.core.dtypes.inference import is_re_compilable

result = is_re_compilable('(')
```

**Output:**
```
PatternError: missing ), unterminated subpattern at position 0
```

**Expected output:** `False`

## Why This Is A Bug

The function's docstring explicitly states:

```python
def is_re_compilable(obj) -> bool:
    """
    Check if the object can be compiled into a regex pattern instance.

    Returns
    -------
    bool
        Whether `obj` can be compiled as a regex pattern.
    """
```

The type annotation and docstring promise that the function returns a boolean value. However, when given an invalid regex pattern (which is a valid string that simply isn't a valid regex), the function raises an exception instead. This violates the function's contract.

Additionally, users calling this function would reasonably expect to use it to check whether a string is a valid regex pattern without needing to wrap it in try/except. The current behavior defeats the purpose of having this check function.

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

The fix is simple: catch `re.error` (which is the base class for all regex compilation errors, including `PatternError`) in addition to `TypeError`. This makes the function's behavior match its documentation.