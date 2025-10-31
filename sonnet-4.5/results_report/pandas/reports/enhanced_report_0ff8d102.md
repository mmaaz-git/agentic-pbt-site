# Bug Report: pandas.errors.AbstractMethodError - AttributeError when using classmethod with instance

**Target**: `pandas.errors.AbstractMethodError`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`AbstractMethodError` crashes with `AttributeError` when initialized with an instance object and `methodtype='classmethod'`, then converted to string. The error accepts this combination in `__init__` but fails when attempting to display the error message.

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

# Run the test
test_abstractmethoderror_valid_methodtype_works()
```

<details>

<summary>
**Failing input**: `methodtype='classmethod'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/48/hypo.py", line 17, in <module>
    test_abstractmethoderror_valid_methodtype_works()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/48/hypo.py", line 5, in test_abstractmethoderror_valid_methodtype_works
    def test_abstractmethoderror_valid_methodtype_works(methodtype):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/48/hypo.py", line 11, in test_abstractmethoderror_valid_methodtype_works
    error_str = str(error)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/errors/__init__.py", line 305, in __str__
    name = self.class_instance.__name__
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'DummyClass' object has no attribute '__name__'. Did you mean: '__ne__'?
Falsifying example: test_abstractmethoderror_valid_methodtype_works(
    methodtype='classmethod',
)
```
</details>

## Reproducing the Bug

```python
import pandas.errors as errors

class DummyClass:
    pass

instance = DummyClass()
error = errors.AbstractMethodError(instance, methodtype='classmethod')
print(str(error))
```

<details>

<summary>
AttributeError when converting to string
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/48/repo.py", line 8, in <module>
    print(str(error))
          ~~~^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/errors/__init__.py", line 305, in __str__
    name = self.class_instance.__name__
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'DummyClass' object has no attribute '__name__'. Did you mean: '__ne__'?
```
</details>

## Why This Is A Bug

This violates expected behavior because the `AbstractMethodError.__init__` method accepts any object as `class_instance` without validation, including instances. However, the `__str__` method assumes that when `methodtype == 'classmethod'`, the `class_instance` parameter is a class object (which has the `__name__` attribute).

The inconsistency creates a scenario where an error object can be successfully constructed but crashes when converted to a string, which is problematic for error handling, logging, and debugging. The parameter name `class_instance` and lack of type hints suggest both classes and instances should be accepted, but the implementation doesn't handle all valid combinations correctly.

While passing an instance for a classmethod is semantically incorrect from a Python perspective (classmethods receive `cls` not `self`), the code should either validate and reject this combination at initialization, or handle it gracefully in the `__str__` method. The current implementation does neither, resulting in an ungraceful crash.

## Relevant Context

The bug occurs in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/errors/__init__.py` at lines 303-308:

```python
def __str__(self) -> str:
    if self.methodtype == "classmethod":
        name = self.class_instance.__name__  # Line 305 - assumes class object
    else:
        name = type(self.class_instance).__name__  # Gets class name from instance
```

The documented examples in the docstring (lines 279-291) show proper usage:
- For classmethods: pass `cls` (the class object)
- For instance methods: pass `self` (the instance)

However, the `__init__` method (lines 294-301) accepts any object without validation, and the parameter documentation doesn't specify this restriction. The parameter name `class_instance` is ambiguous and could reasonably be interpreted as "a class or an instance."

Testing shows that the error works correctly for documented use cases:
- When used with `cls` in a classmethod: Works fine
- When used with `self` in a regular method: Works fine
- When used with an instance and `methodtype='staticmethod'`: Works fine
- When used with an instance and `methodtype='property'`: Works fine
- Only fails with an instance and `methodtype='classmethod'`: Crashes with AttributeError

## Proposed Fix

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