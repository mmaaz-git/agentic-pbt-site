# Bug Report: Cython.Debugger.DebugWriter.is_valid_tag - Inconsistent Validation for str vs EncodedString

**Target**: `Cython.Debugger.DebugWriter.is_valid_tag`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `is_valid_tag` function fails to validate tag names like `.0`, `.123` when passed as regular `str` objects, but correctly rejects them when passed as `EncodedString` objects. This inconsistent behavior violates the documented purpose of filtering out generator argument names.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from Cython.Debugger.DebugWriter import is_valid_tag


@given(st.integers(min_value=0, max_value=999999))
def test_is_valid_tag_decimal_pattern(n):
    name = f".{n}"
    assert is_valid_tag(name) is False
```

**Failing input**: `.0` (or any string matching the pattern `"." + decimal_digits`)

## Reproducing the Bug

```python
from Cython.Debugger.DebugWriter import is_valid_tag
from Cython.Compiler.StringEncoding import EncodedString

print(is_valid_tag('.0'))
print(is_valid_tag('.123'))

print(is_valid_tag(EncodedString('.0')))
print(is_valid_tag(EncodedString('.123')))
```

Output:
```
True
True
False
False
```

## Why This Is A Bug

The function's docstring explicitly states that "Names like '.0' are used internally for arguments to functions creating generator expressions, however they are not identifiers." The function is supposed to filter out such names, but only does so when the input is an `EncodedString`, not when it's a regular `str`. This creates inconsistent behavior where the same logical tag name is considered valid or invalid depending on its type.

## Fix

```diff
def is_valid_tag(name):
    """
    Names like '.0' are used internally for arguments
    to functions creating generator expressions,
    however they are not identifiers.

    See https://github.com/cython/cython/issues/5552
    """
-   if isinstance(name, EncodedString):
-       if name.startswith(".") and name[1:].isdecimal():
-           return False
+   if name.startswith(".") and name[1:].isdecimal():
+       return False
    return True
```