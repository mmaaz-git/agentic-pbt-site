# Bug Report: pandas.errors.AbstractMethodError.__str__ Crashes with AttributeError

**Target**: `pandas.errors.AbstractMethodError.__str__`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

When `AbstractMethodError` is created with `methodtype="classmethod"` but receives an instance instead of a class, calling `str()` on it raises an `AttributeError` because the code tries to access `__name__` on an instance object.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas.errors


class DummyClass:
    pass


@given(st.sampled_from(["method", "classmethod", "staticmethod", "property"]))
def test_abstract_method_error_valid_methodtype(valid_type):
    err = pandas.errors.AbstractMethodError(DummyClass(), methodtype=valid_type)
    assert err.methodtype == valid_type

    msg = str(err)
    assert valid_type in msg
    assert "DummyClass" in msg
    assert "must be defined in the concrete class" in msg
```

**Failing input**: `valid_type='classmethod'`

## Reproducing the Bug

```python
import pandas.errors


class DummyClass:
    pass


instance = DummyClass()
err = pandas.errors.AbstractMethodError(instance, methodtype="classmethod")

try:
    msg = str(err)
    print(f"String representation: {msg}")
except AttributeError as e:
    print(f"AttributeError: {e}")
```

Output:
```
AttributeError: 'DummyClass' object has no attribute '__name__'
```

## Why This Is A Bug

The `__init__` method accepts `class_instance` without validating that it matches the `methodtype`. When `methodtype="classmethod"`, the `__str__` method assumes `class_instance` is a class (and accesses `..__name__`), but when `methodtype="method"`, it assumes it's an instance (and uses `type(class_instance).__name__`).

This creates an inconsistent API where:
- Creating the error succeeds
- But converting it to a string crashes

The docstring examples show the intended usage (passing `cls` for classmethod), but there's no enforcement of this contract.

## Fix

Option 1: Add validation in `__init__` to ensure the right type is passed for each methodtype:

```diff
--- a/pandas/errors/__init__.py
+++ b/pandas/errors/__init__.py
@@ -294,6 +294,11 @@ class AbstractMethodError(NotImplementedError):
     def __init__(self, class_instance, methodtype: str = "method") -> None:
         types = {"method", "classmethod", "staticmethod", "property"}
         if methodtype not in types:
             raise ValueError(
                 f"methodtype must be one of {types}, got {methodtype} instead."
             )
+        if methodtype in ("classmethod", "staticmethod") and not isinstance(class_instance, type):
+            raise TypeError(
+                f"class_instance must be a class (not an instance) when methodtype is {methodtype!r}"
+            )
         self.methodtype = methodtype
         self.class_instance = class_instance
```

Option 2: Make `__str__` more robust to handle both classes and instances:

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

Option 1 is preferred as it enforces the contract more clearly and fails early.