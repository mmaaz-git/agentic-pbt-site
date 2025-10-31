# Bug Report: pandas.errors.AbstractMethodError Crash with classmethod

**Target**: `pandas.errors.AbstractMethodError`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`AbstractMethodError` crashes with `AttributeError` when `methodtype='classmethod'` is used with a class instance, which is the documented usage pattern.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas as pd


class DummyClass:
    pass


@given(st.sampled_from(["method", "classmethod", "staticmethod", "property"]))
def test_abstractmethoderror_valid_methodtypes_should_not_crash(methodtype):
    instance = DummyClass()
    err = pd.errors.AbstractMethodError(instance, methodtype=methodtype)
    error_message = str(err)
    assert isinstance(error_message, str)
    assert len(error_message) > 0
```

**Failing input**: `methodtype='classmethod'`

## Reproducing the Bug

```python
import pandas as pd


class TestClass:
    pass


err = pd.errors.AbstractMethodError(TestClass(), methodtype="classmethod")
print(str(err))
```

**Output:**
```
AttributeError: 'TestClass' object has no attribute '__name__'. Did you mean: '__ne__'?
```

## Why This Is A Bug

The `__str__` method in `AbstractMethodError` assumes that when `methodtype='classmethod'`, the `class_instance` parameter will be a class object (which has `__name__`). However, the documentation and typical usage pass an instance, not a class. When `methodtype='classmethod'`, the code tries to access `self.class_instance.__name__`, but instances don't have a `__name__` attribute, causing a crash.

This violates the principle that all valid `methodtype` values should work without crashing.

## Fix

```diff
--- a/pandas/errors/__init__.py
+++ b/pandas/errors/__init__.py
@@ -302,7 +302,10 @@ class AbstractMethodError(NotImplementedError):

     def __str__(self) -> str:
         if self.methodtype == "classmethod":
-            name = self.class_instance.__name__
+            if isinstance(self.class_instance, type):
+                name = self.class_instance.__name__
+            else:
+                name = type(self.class_instance).__name__
         else:
             name = type(self.class_instance).__name__
         return f"This {self.methodtype} must be defined in the concrete class {name}"
```