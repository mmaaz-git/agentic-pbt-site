# Bug Report: pandas.errors.AbstractMethodError Invalid Methodtype Error Message

**Target**: `pandas.errors.AbstractMethodError.__init__`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `AbstractMethodError.__init__` method has swapped variables in its ValueError message. When an invalid methodtype is provided, the error message incorrectly shows the invalid value where it should show the valid options, and vice versa.

## Property-Based Test

```python
import pandas.errors as pd_errors
from hypothesis import given, strategies as st, assume


@given(st.text(min_size=1))
def test_abstractmethoderror_error_message_property(invalid_methodtype):
    valid_types = {"method", "classmethod", "staticmethod", "property"}
    assume(invalid_methodtype not in valid_types)

    class DummyClass:
        pass

    try:
        pd_errors.AbstractMethodError(DummyClass(), methodtype=invalid_methodtype)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        error_msg = str(e)
        assert str(valid_types) in error_msg, f"Error should mention valid types, got: {error_msg}"
        assert f"got {invalid_methodtype}" in error_msg, f"Error should mention invalid input, got: {error_msg}"
```

**Failing input**: `invalid_methodtype="foo"` (or any string not in the valid set)

## Reproducing the Bug

```python
import pandas.errors as pd_errors


class DummyClass:
    pass


pd_errors.AbstractMethodError(DummyClass(), methodtype="foo")
```

**Current output**:
```
ValueError: methodtype must be one of foo, got {'method', 'classmethod', 'staticmethod', 'property'} instead.
```

**Expected output**:
```
ValueError: methodtype must be one of {'method', 'classmethod', 'staticmethod', 'property'}, got foo instead.
```

## Why This Is A Bug

The error message violates the principle of clear error reporting. It confuses users by showing the invalid value ("foo") where the valid options should be listed, and showing the valid options where the problematic input should be displayed. This makes debugging harder for users who receive this error.

The code documentation and intent clearly show that `methodtype` should be validated against a specific set of valid values, but the error message gets the variables backwards.

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