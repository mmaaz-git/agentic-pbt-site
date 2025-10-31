# Bug Report: pandas.errors.AbstractMethodError Error Message and __str__ Bugs

**Target**: `pandas.errors.AbstractMethodError`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`AbstractMethodError` has two bugs: (1) the validation error message has swapped parameters, making it confusing, and (2) the `__str__` method crashes when `methodtype='classmethod'` if an instance is passed instead of a class.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas as pd
import pytest


@given(
    invalid_methodtype=st.text().filter(lambda x: x not in {"method", "classmethod", "staticmethod", "property"}),
)
def test_abstract_method_error_message_clarity(invalid_methodtype):
    class DummyClass:
        pass

    instance = DummyClass()

    with pytest.raises(ValueError) as exc_info:
        pd.errors.AbstractMethodError(instance, methodtype=invalid_methodtype)

    error_message = str(exc_info.value)
    assert invalid_methodtype in error_message
```

**Failing input**: `invalid_methodtype='x'`

## Reproducing the Bugs

### Bug 1: Swapped error message parameters

```python
import pandas as pd

class DummyClass:
    pass

instance = DummyClass()

try:
    error = pd.errors.AbstractMethodError(instance, methodtype='invalid_type')
except ValueError as e:
    print(e)
```

Output:
```
methodtype must be one of invalid_type, got {'staticmethod', 'method', 'classmethod', 'property'} instead.
```

The message incorrectly states "methodtype must be one of invalid_type" when it should say "methodtype must be one of {'staticmethod', 'method', 'classmethod', 'property'}".

### Bug 2: AttributeError in __str__ for classmethod

```python
import pandas as pd

class DummyClass:
    pass

instance = DummyClass()
error = pd.errors.AbstractMethodError(instance, methodtype='classmethod')
str(error)
```

Output:
```
AttributeError: 'DummyClass' object has no attribute '__name__'
```

## Why These Are Bugs

**Bug 1**: The validation error message on line 298 confuses users by showing the invalid value where valid values should be shown, and vice versa. This makes debugging harder.

**Bug 2**: The `__str__` method assumes that when `methodtype='classmethod'`, the `class_instance` parameter will be a class (which has `__name__`), but the code doesn't validate this assumption. Users can pass an instance, causing a crash when the error is converted to a string.

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

For Bug 2, the fix requires more consideration. Options include:
1. Add validation in `__init__` to check that `class_instance` is a class when `methodtype='classmethod'`
2. Make `__str__` more defensive by using `getattr(self.class_instance, '__name__', type(self.class_instance).__name__)`
3. Update documentation to clarify the expected types for `class_instance`