# Bug Report: pandas.errors.AbstractMethodError - Swapped error message parameters

**Target**: `pandas.errors.AbstractMethodError.__init__`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The validation error message in `AbstractMethodError.__init__` has swapped parameters, showing "methodtype must be one of {invalid_value}, got {valid_types}" instead of the correct order.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas.errors as errors
import pytest

@given(st.text(min_size=1))
def test_abstractmethoderror_invalid_methodtype_message(invalid_methodtype):
    valid_types = {"method", "classmethod", "staticmethod", "property"}
    if invalid_methodtype in valid_types:
        return

    class DummyClass:
        pass

    instance = DummyClass()

    with pytest.raises(ValueError) as exc_info:
        errors.AbstractMethodError(instance, methodtype=invalid_methodtype)

    error_message = str(exc_info.value)

    for valid_type in valid_types:
        assert valid_type in error_message
    assert invalid_methodtype not in f"must be one of {invalid_methodtype}"
```

**Failing input**: Any string not in `{"method", "classmethod", "staticmethod", "property"}`, e.g. `"invalid"`

## Reproducing the Bug

```python
import pandas.errors as errors

class DummyClass:
    pass

try:
    errors.AbstractMethodError(DummyClass(), methodtype='invalid')
except ValueError as e:
    print(e)
```

Output:
```
methodtype must be one of invalid, got {'method', 'property', 'staticmethod', 'classmethod'} instead.
```

Expected:
```
methodtype must be one of {'method', 'property', 'staticmethod', 'classmethod'}, got invalid instead.
```

## Why This Is A Bug

The error message is confusing and incorrect. It says "methodtype must be one of invalid" which is nonsensical - the invalid value should not be presented as what the methodtype must be.

## Fix

```diff
--- a/pandas/errors/__init__.py
+++ b/pandas/errors/__init__.py
@@ -297,7 +297,7 @@ class AbstractMethodError(NotImplementedError):
         types = {"method", "classmethod", "staticmethod", "property"}
         if methodtype not in types:
             raise ValueError(
-                f"methodtype must be one of {methodtype}, got {types} instead."
+                f"methodtype must be one of {types}, got {methodtype} instead."
             )
         self.methodtype = methodtype
         self.class_instance = class_instance
```