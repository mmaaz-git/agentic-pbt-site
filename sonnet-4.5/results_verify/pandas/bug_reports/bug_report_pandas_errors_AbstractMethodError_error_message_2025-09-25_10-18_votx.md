# Bug Report: pandas.errors.AbstractMethodError Error Message Has Swapped Variables

**Target**: `pandas.errors.AbstractMethodError.__init__`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The error message raised when an invalid `methodtype` is passed to `AbstractMethodError` has swapped variable names, making the error message confusing and backwards.

## Property-Based Test

```python
@given(st.text().filter(lambda x: x not in {"method", "classmethod", "staticmethod", "property"}))
def test_abstract_method_error_invalid_methodtype(invalid_methodtype):
    class DummyClass:
        pass

    instance = DummyClass()

    with pytest.raises(ValueError) as exc_info:
        pd_errors.AbstractMethodError(instance, methodtype=invalid_methodtype)

    error_msg = str(exc_info.value)
    assert "methodtype must be one of" in error_msg
```

**Failing input**: `invalid_methodtype='invalid_type'`

## Reproducing the Bug

```python
import pandas.errors as pd_errors


class DummyClass:
    pass


instance = DummyClass()

try:
    error = pd_errors.AbstractMethodError(instance, methodtype="invalid_type")
except ValueError as e:
    print(f"Actual error message: {e}")
    print()
    print("Expected error message:")
    print("  methodtype must be one of {'method', 'classmethod', 'staticmethod', 'property'}, got 'invalid_type' instead.")
```

**Output:**
```
Actual error message: methodtype must be one of invalid_type, got {'staticmethod', 'property', 'classmethod', 'method'} instead.

Expected error message:
  methodtype must be one of {'method', 'classmethod', 'staticmethod', 'property'}, got 'invalid_type' instead.
```

## Why This Is A Bug

The error message on line 298 of `/pandas/errors/__init__.py` has the variable names swapped. It says:
- "methodtype must be one of {the_invalid_input}, got {the_valid_types} instead"

When it should say:
- "methodtype must be one of {the_valid_types}, got {the_invalid_input} instead"

This violates the expected error message format and makes debugging harder.

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