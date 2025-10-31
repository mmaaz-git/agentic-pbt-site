# Bug Report: Cython.Debugger.DebugWriter.is_valid_tag Type-Dependent Behavior

**Target**: `Cython.Debugger.DebugWriter.is_valid_tag`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `is_valid_tag` function exhibits inconsistent behavior: it validates names differently depending on whether the input is a regular `str` or an `EncodedString`, causing it to accept invalid tag names like `.0` when passed as regular strings.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, example
from Cython.Debugger.DebugWriter import is_valid_tag
from Cython.Compiler.StringEncoding import EncodedString


@given(st.text())
@example('.0')
@example('.123')
@example('.999')
@settings(max_examples=1000)
def test_is_valid_tag_consistency(s):
    regular_result = is_valid_tag(s)
    encoded_result = is_valid_tag(EncodedString(s))
    assert regular_result == encoded_result, \
        f"Inconsistent: is_valid_tag({s!r}) = {regular_result}, but is_valid_tag(EncodedString({s!r})) = {encoded_result}"
```

**Failing input**: `.0` (and similar patterns like `.123`, `.999`)

## Reproducing the Bug

```python
from Cython.Debugger.DebugWriter import is_valid_tag
from Cython.Compiler.StringEncoding import EncodedString

test_input = ".0"

result_str = is_valid_tag(test_input)
result_encoded = is_valid_tag(EncodedString(test_input))

print(f'is_valid_tag("{test_input}") = {result_str}')
print(f'is_valid_tag(EncodedString("{test_input}")) = {result_encoded}')

assert result_str == result_encoded
```

## Why This Is A Bug

The function is designed to filter out invalid tag names (like `.0`) that are used internally by Cython for generator expression arguments. However, it only performs this check when the input is an `EncodedString`, not when it's a regular `str`. This creates an API inconsistency where the same logical name is treated differently based on its type.

The current implementation:
```python
def is_valid_tag(name):
    if isinstance(name, EncodedString):
        if name.startswith(".") and name[1:].isdecimal():
            return False
    return True
```

This means `is_valid_tag(".0")` returns `True`, while `is_valid_tag(EncodedString(".0"))` returns `False`, violating the principle of type transparency.

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
+   if name.startswith(".") and len(name) > 1 and name[1:].isdecimal():
+       return False
    return True
```