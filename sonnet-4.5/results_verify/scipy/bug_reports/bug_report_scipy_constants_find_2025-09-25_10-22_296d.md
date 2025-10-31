# Bug Report: scipy.constants.find() Type Validation

**Target**: `scipy.constants.find()`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `find()` function crashes with an unclear `AttributeError` when passed a non-string argument, despite having a type hint that suggests it accepts `str | None`. The function should validate the input type and raise a clear `TypeError`.

## Property-Based Test

```python
import scipy.constants as sc
from hypothesis import given, strategies as st, settings
import pytest


@given(st.integers())
@settings(max_examples=100)
def test_find_with_integer_should_handle_gracefully(num):
    result = sc.find(num)
    assert isinstance(result, list)
```

**Failing input**: `num=0` (or any integer)

## Reproducing the Bug

```python
import scipy.constants as sc

sc.find(123)
```

**Output:**
```
AttributeError: 'int' object has no attribute 'lower'
```

## Why This Is A Bug

1. The function signature includes a type hint `sub: str | None = None`, indicating it expects a string or None
2. When passed a non-string value (e.g., an integer), it crashes with `AttributeError` instead of validating the input
3. The error message "object has no attribute 'lower'" is unclear and doesn't indicate the actual problem (wrong type)
4. Good API design requires explicit input validation with clear error messages

## Fix

```diff
--- a/scipy/constants/_codata.py
+++ b/scipy/constants/_codata.py
@@ -2248,6 +2248,8 @@ def find(sub: str | None = None, disp: bool = False) -> Any:
     """
     if sub is None:
         result = list(_current_constants.keys())
     else:
+        if not isinstance(sub, str):
+            raise TypeError(f"sub must be a string or None, got {type(sub).__name__}")
         result = [key for key in _current_constants
                   if sub.lower() in key.lower()]
```