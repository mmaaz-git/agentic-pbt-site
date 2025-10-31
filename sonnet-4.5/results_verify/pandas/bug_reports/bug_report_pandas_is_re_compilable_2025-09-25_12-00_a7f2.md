# Bug Report: pandas.api.types.is_re_compilable Raises Exception Instead of Returning False

**Target**: `pandas.api.types.is_re_compilable`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`is_re_compilable()` raises `re.PatternError` for invalid regex patterns instead of returning `False` as documented. The function only catches `TypeError` but not `re.error`, violating its contract to always return a bool.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas.api.types as pat

@given(st.text(min_size=1, max_size=10))
def test_is_re_compilable_returns_bool(s):
    """is_re_compilable should always return a bool, never raise exceptions"""
    result = pat.is_re_compilable(s)
    assert isinstance(result, bool), f"is_re_compilable should return bool"
```

**Failing input**: `')'` (and other invalid regex patterns like `'?'`, `'*'`, `'+'`, `'('`, `'['`, `'\\'`)

## Reproducing the Bug

```python
import pandas.api.types as pat

print(pat.is_re_compilable(')'))
```

Expected: `False`
Actual: Raises `re.PatternError: unbalanced parenthesis at position 0`

## Why This Is A Bug

The function's docstring explicitly states it returns a `bool` and provides an example showing it should return `False` for non-compilable inputs. However, invalid regex patterns raise `re.PatternError` instead of being caught and returning `False`.

The current implementation only catches `TypeError`, but `re.compile()` raises `re.error` (or its subclass `re.PatternError` in Python 3.13+) for invalid patterns.

## Fix

```diff
--- a/pandas/core/dtypes/inference.py
+++ b/pandas/core/dtypes/inference.py
@@ -184,7 +184,7 @@ def is_re_compilable(obj) -> bool:
     """
     try:
         re.compile(obj)
-    except TypeError:
+    except (TypeError, re.error):
         return False
     else:
         return True
```