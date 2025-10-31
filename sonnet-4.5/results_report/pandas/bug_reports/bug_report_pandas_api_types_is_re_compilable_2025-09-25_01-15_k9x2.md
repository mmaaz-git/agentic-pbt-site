# Bug Report: pandas.api.types.is_re_compilable Invalid Pattern Handling

**Target**: `pandas.api.types.is_re_compilable`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`is_re_compilable()` raises `re.PatternError` instead of returning `False` for invalid regex patterns, violating its documented contract of always returning a boolean.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas.api.types as pat

@given(st.text(min_size=1, max_size=50))
def test_is_re_compilable_returns_bool(pattern):
    result = pat.is_re_compilable(pattern)
    assert isinstance(result, bool), \
        f"is_re_compilable should always return bool, got exception for {repr(pattern)}"
```

**Failing inputs**: `')'`, `'?'`, `'*'`, `'('`, `'['`

## Reproducing the Bug

```python
import pandas.api.types as pat

result = pat.is_re_compilable(')')
```

**Expected**: `False` (the pattern cannot be compiled)
**Actual**: Raises `re.PatternError: unbalanced parenthesis at position 0`

## Why This Is A Bug

The function's type signature is `is_re_compilable(obj) -> bool`, and its docstring states:

> Check if the object can be compiled into a regex pattern instance.
> Returns: bool - Whether `obj` can be compiled as a regex pattern.

This contract promises a boolean return value for all inputs. However, the function raises `re.PatternError` for malformed regex patterns, violating this contract. Users cannot safely use this function without wrapping it in try-except, defeating its purpose as a boolean check function.

## Fix

```diff
--- a/pandas/core/dtypes/inference.py
+++ b/pandas/core/dtypes/inference.py
@@ -185,7 +185,7 @@ def is_re_compilable(obj) -> bool:
     """
     try:
         re.compile(obj)
-    except TypeError:
+    except (TypeError, re.PatternError):
         return False
     else:
         return True
```

Alternatively, catch `Exception` to handle all possible compilation errors:

```diff
--- a/pandas/core/dtypes/inference.py
+++ b/pandas/core/dtypes/inference.py
@@ -185,7 +185,7 @@ def is_re_compilable(obj) -> bool:
     """
     try:
         re.compile(obj)
-    except TypeError:
+    except Exception:
         return False
     else:
         return True
```