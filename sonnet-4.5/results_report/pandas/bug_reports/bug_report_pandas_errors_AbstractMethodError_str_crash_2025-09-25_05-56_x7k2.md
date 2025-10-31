# Bug Report: pandas.errors.AbstractMethodError Crashes on __str__ with classmethod

**Target**: `pandas.errors.AbstractMethodError.__str__`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

When `AbstractMethodError` is created with `methodtype="classmethod"` and an instance object (rather than a class), calling `str()` on the error raises an `AttributeError` because the code assumes `class_instance` will be a class with a `__name__` attribute.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas.errors as pd_errors


class DummyClass:
    pass


@given(st.sampled_from(["method", "classmethod", "staticmethod", "property"]))
def test_abstract_method_error_valid_methodtype(valid_methodtype):
    dummy = DummyClass()
    error = pd_errors.AbstractMethodError(dummy, methodtype=valid_methodtype)
    assert error.methodtype == valid_methodtype
    assert error.class_instance is dummy

    error_message = str(error)
    assert valid_methodtype in error_message
    assert "DummyClass" in error_message
```

**Failing input**: `valid_methodtype='classmethod'` (with an instance object)

## Reproducing the Bug

```python
import pandas.errors as pd_errors


class DummyClass:
    pass


dummy = DummyClass()
error = pd_errors.AbstractMethodError(dummy, methodtype="classmethod")

str(error)
```

**Output**:
```
AttributeError: 'DummyClass' object has no attribute '__name__'
```

## Why This Is A Bug

The `__str__` method at line 304-305 assumes that when `methodtype == "classmethod"`, the `class_instance` will be a class (which has `__name__`), but the `__init__` method accepts any object as `class_instance` without validation. The docstring examples show that classmethods should receive a class, but this is not enforced.

There are two possible interpretations:
1. The `__init__` should validate that for `methodtype="classmethod"`, `class_instance` must be a class
2. The `__str__` should handle both classes and instances gracefully

Looking at the intended usage from the docstring (lines 281-282 and 284), it appears classmethods are expected to receive the class itself, not an instance. However, users might reasonably pass an instance and expect the error to extract the class from it.

## Fix

Option 1: Make `__str__` more robust by handling both classes and instances:

```diff
--- a/pandas/errors/__init__.py
+++ b/pandas/errors/__init__.py
@@ -302,7 +302,10 @@ class AbstractMethodError(NotImplementedError):

     def __str__(self) -> str:
         if self.methodtype == "classmethod":
-            name = self.class_instance.__name__
+            if hasattr(self.class_instance, '__name__'):
+                name = self.class_instance.__name__
+            else:
+                name = type(self.class_instance).__name__
         else:
             name = type(self.class_instance).__name__
         return f"This {self.methodtype} must be defined in the concrete class {name}"
```

Option 2: Validate in `__init__` that classmethods receive a class:

```diff
--- a/pandas/errors/__init__.py
+++ b/pandas/errors/__init__.py
@@ -298,6 +298,9 @@ class AbstractMethodError(NotImplementedError):
                 f"methodtype must be one of {types}, got {methodtype} instead."
             )
         self.methodtype = methodtype
+        if methodtype == "classmethod" and not isinstance(class_instance, type):
+            raise TypeError(
+                "class_instance must be a class (not an instance) when methodtype='classmethod'"
+            )
         self.class_instance = class_instance

     def __str__(self) -> str:
```

Option 1 is recommended as it's more user-friendly and defensive.