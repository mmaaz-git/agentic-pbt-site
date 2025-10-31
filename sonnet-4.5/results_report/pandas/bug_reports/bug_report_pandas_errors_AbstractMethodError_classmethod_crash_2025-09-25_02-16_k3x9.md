# Bug Report: pandas.errors.AbstractMethodError - AttributeError when using classmethod with instance

**Target**: `pandas.errors.AbstractMethodError`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`AbstractMethodError` crashes with `AttributeError` when initialized with an instance and `methodtype='classmethod'`, then converted to string.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas.errors as errors

@given(st.sampled_from(["method", "classmethod", "staticmethod", "property"]))
def test_abstractmethoderror_valid_methodtype_works(methodtype):
    class DummyClass:
        pass

    instance = DummyClass()
    error = errors.AbstractMethodError(instance, methodtype=methodtype)
    error_str = str(error)

    assert methodtype in error_str
    assert "must be defined in the concrete class" in error_str
```

**Failing input**: `methodtype='classmethod'` with any instance object

## Reproducing the Bug

```python
import pandas.errors as errors

class DummyClass:
    pass

instance = DummyClass()
error = errors.AbstractMethodError(instance, methodtype='classmethod')
str(error)
```

Output:
```
AttributeError: 'DummyClass' object has no attribute '__name__'
```

## Why This Is A Bug

The `__str__` method assumes that when `methodtype == 'classmethod'`, the `class_instance` parameter is a class object (which has `__name__`), but the code accepts instance objects in `__init__`. This creates an inconsistent state where the error can be constructed but cannot be converted to a string.

According to the docstring, users can pass either a class or an instance, but the implementation doesn't handle all combinations correctly.

## Fix

```diff
--- a/pandas/errors/__init__.py
+++ b/pandas/errors/__init__.py
@@ -302,7 +302,7 @@ class AbstractMethodError(NotImplementedError):
         self.class_instance = class_instance

     def __str__(self) -> str:
-        if self.methodtype == "classmethod":
+        if self.methodtype == "classmethod" and isinstance(self.class_instance, type):
             name = self.class_instance.__name__
         else:
             name = type(self.class_instance).__name__
```