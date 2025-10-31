# Bug Report: pandas.errors.AbstractMethodError Crashes When Converting to String with methodtype='classmethod'

**Target**: `pandas.errors.AbstractMethodError.__str__`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`AbstractMethodError.__str__` crashes with `AttributeError` when `methodtype='classmethod'` and `class_instance` is an instance rather than a class, because it tries to access `__name__` attribute on an instance object.

## Property-Based Test

```python
@given(st.sampled_from(["method", "classmethod", "staticmethod", "property"]))
def test_abstract_method_error_valid_methodtype(valid_methodtype):
    class DummyClass:
        pass

    instance = DummyClass()
    error = pd_errors.AbstractMethodError(instance, methodtype=valid_methodtype)

    assert error.methodtype == valid_methodtype
    assert error.class_instance is instance

    error_str = str(error)
    assert isinstance(error_str, str)
    assert valid_methodtype in error_str
    assert "DummyClass" in error_str
```

**Failing input**: `valid_methodtype='classmethod'`

## Reproducing the Bug

```python
import pandas.errors as pd_errors


class DummyClass:
    pass


instance = DummyClass()

error = pd_errors.AbstractMethodError(instance, methodtype="classmethod")

print("Error created successfully")

try:
    error_str = str(error)
    print(f"Error string: {error_str}")
except AttributeError as e:
    print(f"CRASH: {e}")
```

**Output:**
```
Error created successfully
CRASH: 'DummyClass' object has no attribute '__name__'
```

## Why This Is A Bug

The `__init__` method accepts any object as `class_instance`, but the `__str__` method (line 305) assumes that when `methodtype == "classmethod"`, the `class_instance` is a class object (which has `__name__`), not an instance.

This violates the Liskov Substitution Principle: if the `__init__` accepts an instance for any `methodtype`, then `__str__` should handle it gracefully for all `methodtype` values.

The docstring example (lines 280-287) shows that for classmethods, a class should be passed:
```python
raise pd.errors.AbstractMethodError(cls, methodtype="classmethod")
```

But there's no validation in `__init__` to enforce this, leading to a crash later when the error is converted to a string.

## Fix

Add validation in `__init__` to ensure proper usage:

```diff
--- a/pandas/errors/__init__.py
+++ b/pandas/errors/__init__.py
@@ -293,6 +293,12 @@ class AbstractMethodError(NotImplementedError):

     def __init__(self, class_instance, methodtype: str = "method") -> None:
         types = {"method", "classmethod", "staticmethod", "property"}
         if methodtype not in types:
             raise ValueError(
                 f"methodtype must be one of {types}, got {methodtype} instead."
             )
+        if methodtype == "classmethod" and not isinstance(class_instance, type):
+            raise TypeError(
+                f"When methodtype='classmethod', class_instance must be a class, "
+                f"not {type(class_instance).__name__}"
+            )
         self.methodtype = methodtype
         self.class_instance = class_instance
```