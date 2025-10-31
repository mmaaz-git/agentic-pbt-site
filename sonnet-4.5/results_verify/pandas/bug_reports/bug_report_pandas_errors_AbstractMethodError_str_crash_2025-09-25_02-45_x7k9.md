# Bug Report: pandas.errors.AbstractMethodError Crashes on str() with Incorrect Usage

**Target**: `pandas.errors.AbstractMethodError`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `AbstractMethodError.__str__` method crashes with an `AttributeError` when `methodtype="classmethod"` and the user passes an instance instead of a class. This should either be validated during initialization or handled gracefully in `__str__`.

## Property-Based Test

```python
import pandas.errors
from hypothesis import given, strategies as st
import pytest


class SampleClass:
    pass


@given(st.sampled_from(["classmethod"]))
def test_abstractmethoderror_str_crashes_with_instance_for_classmethod(methodtype):
    instance = SampleClass()
    error = pandas.errors.AbstractMethodError(instance, methodtype=methodtype)

    with pytest.raises(AttributeError, match="'SampleClass' object has no attribute '__name__'"):
        str(error)
```

**Failing input**: Any instance object when `methodtype="classmethod"`

## Reproducing the Bug

```python
import pandas.errors


class SampleClass:
    pass


instance = SampleClass()
error = pandas.errors.AbstractMethodError(instance, methodtype="classmethod")

try:
    message = str(error)
    print(f"Message: {message}")
except AttributeError as e:
    print(f"Crashed with AttributeError: {e}")
```

Output:
```
Crashed with AttributeError: 'SampleClass' object has no attribute '__name__'. Did you mean: '__ne__'?
```

## Why This Is A Bug

While the documentation shows that `classmethod` should be used with a class (not an instance), the code doesn't validate this assumption. When a user makes this mistake, they get a cryptic `AttributeError` instead of a clear error message explaining what went wrong.

The issue is in the `__str__` method (lines 303-308):

```python
def __str__(self) -> str:
    if self.methodtype == "classmethod":
        name = self.class_instance.__name__  # Assumes class_instance is a class
    else:
        name = type(self.class_instance).__name__  # Works for instances
    return f"This {self.methodtype} must be defined in the concrete class {name}"
```

## Fix

Option 1: Validate during initialization (stricter):

```diff
--- a/pandas/errors/__init__.py
+++ b/pandas/errors/__init__.py
@@ -294,7 +294,12 @@ class AbstractMethodError(NotImplementedError):
     def __init__(self, class_instance, methodtype: str = "method") -> None:
         types = {"method", "classmethod", "staticmethod", "property"}
         if methodtype not in types:
             raise ValueError(
                 f"methodtype must be one of {types}, got {methodtype} instead."
             )
+        if methodtype == "classmethod" and not isinstance(class_instance, type):
+            raise TypeError(
+                f"For methodtype='classmethod', class_instance must be a class, "
+                f"not an instance of {type(class_instance).__name__}"
+            )
         self.methodtype = methodtype
         self.class_instance = class_instance
```

Option 2: Handle gracefully in `__str__` (more lenient):

```diff
--- a/pandas/errors/__init__.py
+++ b/pandas/errors/__init__.py
@@ -303,7 +303,10 @@ class AbstractMethodError(NotImplementedError):
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

Option 1 is recommended as it catches the error earlier and provides clearer feedback.