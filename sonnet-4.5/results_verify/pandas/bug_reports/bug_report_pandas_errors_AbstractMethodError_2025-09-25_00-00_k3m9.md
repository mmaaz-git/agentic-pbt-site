# Bug Report: pandas.errors.AbstractMethodError Swapped Error Message

**Target**: `pandas.errors.AbstractMethodError.__init__`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `AbstractMethodError.__init__` method has swapped values in its error message when an invalid `methodtype` is provided. The message says "methodtype must be one of <invalid_value>, got <valid_types> instead" when it should say "methodtype must be one of <valid_types>, got <invalid_value> instead".

## Property-Based Test

```python
import pytest
from hypothesis import given, strategies as st
import pandas.errors as pd_errors


@given(st.text(min_size=1).filter(lambda x: x not in {"method", "classmethod", "staticmethod", "property"}))
def test_abstract_method_error_swapped_values_bug(invalid_methodtype):
    """
    BUG: AbstractMethodError has swapped values in its error message.

    The error message currently says:
        "methodtype must be one of <invalid_value>, got <valid_types> instead."

    But it should say:
        "methodtype must be one of <valid_types>, got <invalid_value> instead."
    """
    class DummyClass:
        pass

    instance = DummyClass()

    with pytest.raises(ValueError) as exc_info:
        pd_errors.AbstractMethodError(instance, methodtype=invalid_methodtype)

    error_msg = str(exc_info.value)

    if "must be one of" in error_msg and "got" in error_msg:
        parts = error_msg.split("got")
        first_part = parts[0]
        second_part = parts[1] if len(parts) > 1 else ""

        valid_types_str = "{'method', 'classmethod', 'staticmethod', 'property'}"

        assert valid_types_str in first_part or all(t in first_part for t in ["method", "classmethod"]), \
            f"Valid types should appear in 'must be one of' clause, but got: {error_msg}"

        assert invalid_methodtype in second_part, \
            f"Invalid value should appear after 'got', but got: {error_msg}"
```

**Failing input**: `invalid_methodtype='0'` (or any string not in the valid set)

## Reproducing the Bug

```python
import pandas.errors as pd_errors


class DummyClass:
    pass


instance = DummyClass()

try:
    pd_errors.AbstractMethodError(instance, methodtype="invalid")
except ValueError as e:
    print(f"Actual:   {e}")
    print("Expected: methodtype must be one of {'method', 'classmethod', 'staticmethod', 'property'}, got invalid instead.")
```

**Actual output:**
```
methodtype must be one of invalid, got {'method', 'classmethod', 'staticmethod', 'property'} instead.
```

**Expected output:**
```
methodtype must be one of {'method', 'classmethod', 'staticmethod', 'property'}, got invalid instead.
```

## Why This Is A Bug

The error message is confusing and unhelpful because it presents the information backwards:
- It says "methodtype must be one of invalid" - which makes no sense
- It says "got {'method', 'classmethod', 'staticmethod', 'property'}" - which are the valid values, not what was received

This violates the API contract for error messages, which should clearly communicate what was expected vs what was received. Standard error message format is "expected X, got Y" not "expected Y, got X".

## Fix

```diff
--- a/pandas/errors/__init__.py
+++ b/pandas/errors/__init__.py
@@ -295,7 +295,7 @@ class AbstractMethodError(NotImplementedError):
         types = {"method", "classmethod", "staticmethod", "property"}
         if methodtype not in types:
             raise ValueError(
-                f"methodtype must be one of {methodtype}, got {types} instead."
+                f"methodtype must be one of {types}, got {methodtype} instead."
             )
         self.methodtype = methodtype
         self.class_instance = class_instance
```