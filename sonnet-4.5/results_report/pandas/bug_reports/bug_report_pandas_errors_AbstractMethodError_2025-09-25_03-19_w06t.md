# Bug Report: pandas.errors.AbstractMethodError Invalid Methodtype Error Message

**Target**: `pandas.errors.AbstractMethodError`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `AbstractMethodError.__init__` method raises a `ValueError` when an invalid `methodtype` is provided, but the error message has swapped variables - it shows the invalid input where valid types should be listed, and shows valid types where the invalid input should be shown.

## Property-Based Test

```python
import pytest
from hypothesis import given, strategies as st, assume
import pandas.errors as pe


@given(st.text(min_size=1))
def test_abstractmethoderror_error_message_shows_correct_values(invalid_methodtype):
    """
    Property: When AbstractMethodError raises ValueError for invalid methodtype,
    the error message should correctly display:
    1. The set of valid types in the "must be one of X" part
    2. The invalid value provided in the "got Y instead" part

    This is a basic contract: error messages should accurately describe what
    went wrong by showing valid options and the invalid input.
    """
    valid_types = {"method", "classmethod", "staticmethod", "property"}

    assume(invalid_methodtype not in valid_types)

    class DummyClass:
        pass
    obj = DummyClass()

    with pytest.raises(ValueError) as exc_info:
        pe.AbstractMethodError(obj, methodtype=invalid_methodtype)

    error_message = str(exc_info.value)

    assert "methodtype must be one of" in error_message

    msg_parts = error_message.split("got")
    assert len(msg_parts) == 2

    first_part = msg_parts[0]
    second_part = msg_parts[1]

    for valid_type in valid_types:
        assert valid_type in first_part, \
            f"Valid type '{valid_type}' should appear in first part (before 'got'), but got: {error_message}"

    assert invalid_methodtype in second_part, \
        f"Invalid type '{invalid_methodtype}' should appear in second part (after 'got'), but got: {error_message}"
```

**Failing input**: `'0'`

## Reproducing the Bug

```python
import pandas.errors as pe


class DummyClass:
    pass


obj = DummyClass()

try:
    pe.AbstractMethodError(obj, methodtype='0')
except ValueError as e:
    print(f"Actual:   {e}")
    print(f"Expected: methodtype must be one of {{'method', 'classmethod', 'staticmethod', 'property'}}, got 0 instead.")
```

**Output:**
```
Actual:   methodtype must be one of 0, got {'property', 'staticmethod', 'method', 'classmethod'} instead.
Expected: methodtype must be one of {'method', 'classmethod', 'staticmethod', 'property'}, got 0 instead.
```

## Why This Is A Bug

The error message is confusing and misleading. It tells users that "methodtype must be one of 0" (the invalid value they provided) and "got {'property', 'staticmethod', 'method', 'classmethod'}" (the valid options), which is backwards. This violates the API contract that error messages should clearly communicate what went wrong.

## Fix

```diff
--- a/pandas/errors/__init__.py
+++ b/pandas/errors/__init__.py
@@ -XX,7 +XX,7 @@ class AbstractMethodError(NotImplementedError):
     def __init__(self, class_instance, methodtype: str = "method") -> None:
         types = {"method", "classmethod", "staticmethod", "property"}
         if methodtype not in types:
             raise ValueError(
-                f"methodtype must be one of {methodtype}, got {types} instead."
+                f"methodtype must be one of {types}, got {methodtype} instead."
             )
         self.methodtype = methodtype
         self.class_instance = class_instance
```