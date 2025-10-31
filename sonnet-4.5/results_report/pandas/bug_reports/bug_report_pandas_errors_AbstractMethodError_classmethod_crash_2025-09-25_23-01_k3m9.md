# Bug Report: pandas.errors.AbstractMethodError Crashes with Instance and methodtype='classmethod'

**Target**: `pandas.errors.AbstractMethodError`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`AbstractMethodError.__str__` crashes with `AttributeError` when `methodtype='classmethod'` is used with a class instance instead of a class object, because it incorrectly assumes `class_instance` will have a `__name__` attribute.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas.errors


@given(st.sampled_from(['method', 'classmethod', 'staticmethod', 'property']))
def test_abstract_method_error_valid_types(methodtype):
    class DummyClass:
        pass

    instance = DummyClass()
    error = pandas.errors.AbstractMethodError(instance, methodtype=methodtype)

    error_str = str(error)
    assert methodtype in error_str
    assert 'DummyClass' in error_str
```

**Failing input**: `methodtype='classmethod'`

## Reproducing the Bug

```python
import pandas.errors


class MyClass:
    pass


instance = MyClass()
error = pandas.errors.AbstractMethodError(instance, methodtype='classmethod')

print(str(error))
```

**Output:**
```
AttributeError: 'MyClass' object has no attribute '__name__'
```

## Why This Is A Bug

While the docstring example shows passing `cls` (the class) for classmethods, there's no validation preventing users from passing an instance. When an instance is passed with `methodtype='classmethod'`, the `__str__` method crashes instead of gracefully handling it. The code should either:

1. Validate that a class (not instance) is passed when `methodtype='classmethod'`, OR
2. Handle both cases in `__str__` by using `type(class_instance).__name__` consistently

## Fix

The simplest fix is to use `type(class_instance).__name__` for all cases except when we know we have a class:

```diff
--- a/pandas/errors/__init__.py
+++ b/pandas/errors/__init__.py
@@ -302,7 +302,11 @@ class AbstractMethodError(NotImplementedError):
         self.class_instance = class_instance

     def __str__(self) -> str:
-        if self.methodtype == "classmethod":
+        if self.methodtype == "classmethod" and hasattr(
+            self.class_instance, "__name__"
+        ):
             name = self.class_instance.__name__
         else:
             name = type(self.class_instance).__name__
         return f"This {self.methodtype} must be defined in the concrete class {name}"
```