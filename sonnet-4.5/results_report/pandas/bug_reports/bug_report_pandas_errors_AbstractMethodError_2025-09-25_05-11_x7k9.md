# Bug Report: pandas.errors.AbstractMethodError Crashes with classmethod

**Target**: `pandas.errors.AbstractMethodError`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`AbstractMethodError` crashes with `AttributeError` when `methodtype='classmethod'` is used with an instance object instead of a class object. The `__str__` method incorrectly assumes `class_instance` is always a class when `methodtype='classmethod'`, but it can also be an instance.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas.errors

@given(st.sampled_from(['method', 'classmethod', 'staticmethod', 'property']))
def test_abstractmethoderror_valid_methodtype(methodtype):
    """
    Property: AbstractMethodError should accept all documented valid methodtypes
    without crashing.
    """
    class DummyClass:
        pass

    instance = DummyClass()
    error = pandas.errors.AbstractMethodError(instance, methodtype=methodtype)
    error_str = str(error)

    assert methodtype in error_str
    assert "DummyClass" in error_str
```

**Failing input**: `methodtype='classmethod'`

## Reproducing the Bug

```python
import pandas as pd

class MyClass:
    pass

instance = MyClass()
error = pd.errors.AbstractMethodError(instance, methodtype='classmethod')
print(str(error))
```

**Output:**
```
AttributeError: 'MyClass' object has no attribute '__name__'. Did you mean: '__ne__'?
```

## Why This Is A Bug

The `AbstractMethodError.__str__` method assumes that when `methodtype='classmethod'`, the `class_instance` parameter is a class object with a `__name__` attribute. However, the constructor accepts both class objects and instance objects for all method types. When an instance is passed with `methodtype='classmethod'`, calling `str()` on the error crashes.

This violates the contract that exception objects should always be convertible to strings for display purposes.

## Fix

```diff
--- a/pandas/errors/__init__.py
+++ b/pandas/errors/__init__.py
@@ -302,7 +302,7 @@ class AbstractMethodError(NotImplementedError):

     def __str__(self) -> str:
         if self.methodtype == "classmethod":
-            name = self.class_instance.__name__
+            name = self.class_instance.__name__ if isinstance(self.class_instance, type) else type(self.class_instance).__name__
         else:
             name = type(self.class_instance).__name__
         return f"This {self.methodtype} must be defined in the concrete class {name}"
```