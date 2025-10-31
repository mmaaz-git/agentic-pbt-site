# Bug Report: AbstractMethodError Crashes When methodtype='classmethod' Used With Instance

**Target**: `pandas.errors.AbstractMethodError.__str__`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

When `AbstractMethodError` is created with `methodtype='classmethod'` but passed an instance instead of a class, calling `str()` on the error raises an `AttributeError` because it tries to access `__name__` on an instance.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas.errors

@given(st.sampled_from(["method", "classmethod", "staticmethod", "property"]))
def test_abstract_method_error_str_works_for_all_methodtypes(methodtype):
    class TestClass:
        pass

    instance = TestClass()
    error = pandas.errors.AbstractMethodError(instance, methodtype=methodtype)

    error_str = str(error)
    assert isinstance(error_str, str)
    assert len(error_str) > 0
```

**Failing input**: `methodtype='classmethod'`

## Reproducing the Bug

```python
import pandas.errors

class MyClass:
    pass

error = pandas.errors.AbstractMethodError(MyClass(), methodtype="classmethod")

try:
    message = str(error)
except AttributeError as e:
    print(f"AttributeError: {e}")
```

**Output:**
```
AttributeError: 'MyClass' object has no attribute '__name__'. Did you mean: '__ne__'?
```

## Why This Is A Bug

The `__str__` method assumes that when `methodtype='classmethod'`, the `class_instance` parameter is a class (which has `__name__`), but the `__init__` method accepts any object. When an instance is passed with `methodtype='classmethod'`, the `__str__` method crashes.

This violates the API contract - if `__init__` accepts an argument combination, `__str__` should work on the resulting object.

## Fix

The fix should either:
1. Validate in `__init__` that `class_instance` is a class when `methodtype='classmethod'`, OR
2. Handle instances gracefully in `__str__` by using `type(class_instance).__name__` for all cases

**Option 2 (simpler):**

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

**Option 1 (stricter validation):**

```diff
--- a/pandas/errors/__init__.py
+++ b/pandas/errors/__init__.py
@@ -294,6 +294,8 @@ class AbstractMethodError(NotImplementedError):
     def __init__(self, class_instance, methodtype: str = "method") -> None:
         types = {"method", "classmethod", "staticmethod", "property"}
         if methodtype not in types:
             raise ValueError(
                 f"methodtype must be one of {types}, got {methodtype} instead."
             )
+        if methodtype == "classmethod" and not isinstance(class_instance, type):
+            raise TypeError("class_instance must be a class when methodtype='classmethod'")
         self.methodtype = methodtype
         self.class_instance = class_instance
```