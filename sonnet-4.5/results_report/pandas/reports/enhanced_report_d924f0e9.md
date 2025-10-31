# Bug Report: pandas.errors.AbstractMethodError Crashes When Instance Passed with methodtype='classmethod'

**Target**: `pandas.errors.AbstractMethodError`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `AbstractMethodError.__str__` method crashes with an `AttributeError` when an instance object is passed with `methodtype='classmethod'`, because it incorrectly assumes the `class_instance` parameter will have a `__name__` attribute (which only classes have, not instances).

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


if __name__ == "__main__":
    test_abstract_method_error_valid_types()
```

<details>

<summary>
**Failing input**: `methodtype='classmethod'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 19, in <module>
    test_abstract_method_error_valid_types()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 6, in test_abstract_method_error_valid_types
    def test_abstract_method_error_valid_types(methodtype):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 13, in test_abstract_method_error_valid_types
    error_str = str(error)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/errors/__init__.py", line 305, in __str__
    name = self.class_instance.__name__
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'DummyClass' object has no attribute '__name__'. Did you mean: '__ne__'?
Falsifying example: test_abstract_method_error_valid_types(
    methodtype='classmethod',
)
```
</details>

## Reproducing the Bug

```python
import pandas.errors


class MyClass:
    pass


instance = MyClass()
error = pandas.errors.AbstractMethodError(instance, methodtype='classmethod')

print(str(error))
```

<details>

<summary>
AttributeError when calling str() on AbstractMethodError
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/28/repo.py", line 11, in <module>
    print(str(error))
          ~~~^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/errors/__init__.py", line 305, in __str__
    name = self.class_instance.__name__
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'MyClass' object has no attribute '__name__'. Did you mean: '__ne__'?
```
</details>

## Why This Is A Bug

This violates expected behavior in multiple ways:

1. **Unhandled Exception in Error Handling**: The `__str__` method of an exception class should never crash. It's frequently called implicitly during debugging, logging, or error reporting, and should always return a string gracefully.

2. **Inconsistent API Behavior**: The other `methodtype` values ('method', 'staticmethod', 'property') all work correctly with instance objects. Only 'classmethod' crashes, creating an unexpected inconsistency.

3. **Ambiguous Parameter Naming**: The parameter is named `class_instance`, suggesting it can accept either a class OR an instance. Nothing in the documentation explicitly prohibits passing an instance.

4. **No Input Validation**: The constructor accepts the instance without validation, but then the `__str__` method crashes when trying to use it. If instances are not allowed for classmethods, this should be validated earlier.

5. **Documentation Gap**: While the docstring examples show passing `cls` for classmethods, there's no explicit warning that passing an instance will cause a crash.

## Relevant Context

The bug occurs in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/errors/__init__.py` at lines 304-308:

```python
def __str__(self) -> str:
    if self.methodtype == "classmethod":
        name = self.class_instance.__name__  # Line 305 - assumes class_instance is a class
    else:
        name = type(self.class_instance).__name__
    return f"This {self.methodtype} must be defined in the concrete class {name}"
```

Classes in Python have a `__name__` attribute, but instances do not. The code assumes that when `methodtype='classmethod'`, the `class_instance` parameter will be a class object (with `__name__`), not an instance.

Additionally, there's a minor documentation error on line 291 where the error message in the example incorrectly says "classmethod" when it should say "method" for the regular method example.

Documentation link: https://pandas.pydata.org/docs/reference/api/pandas.errors.AbstractMethodError.html

## Proposed Fix

```diff
--- a/pandas/errors/__init__.py
+++ b/pandas/errors/__init__.py
@@ -302,7 +302,10 @@ class AbstractMethodError(NotImplementedError):
         self.class_instance = class_instance

     def __str__(self) -> str:
-        if self.methodtype == "classmethod":
+        if self.methodtype == "classmethod" and hasattr(
+            self.class_instance, "__name__"
+        ):
+            # Use __name__ directly only if it's actually a class
             name = self.class_instance.__name__
         else:
             name = type(self.class_instance).__name__
```