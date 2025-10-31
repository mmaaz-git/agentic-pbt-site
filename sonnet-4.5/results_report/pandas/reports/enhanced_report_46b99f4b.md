# Bug Report: pandas.errors.AbstractMethodError Crashes When Converting to String with classmethod Type

**Target**: `pandas.errors.AbstractMethodError`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`AbstractMethodError` crashes with `AttributeError` when attempting to convert to string if `methodtype='classmethod'` is used with an instance object instead of a class object, violating the fundamental Python contract that exceptions should always be convertible to strings.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas.errors

@given(st.sampled_from(['method', 'classmethod', 'staticmethod', 'property']))
def test_abstractmethoderror_valid_methodtype(methodtype):
    """
    Property: AbstractMethodError should accept all documented valid methodtypes
    without crashing.
    """
    class DummyClass:
        pass

    instance = DummyClass()
    error = pandas.errors.AbstractMethodError(instance, methodtype=methodtype)
    error_str = str(error)

    assert methodtype in error_str
    assert "DummyClass" in error_str

if __name__ == "__main__":
    test_abstractmethoderror_valid_methodtype()
```

<details>

<summary>
**Failing input**: `methodtype='classmethod'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 21, in <module>
    test_abstractmethoderror_valid_methodtype()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 5, in test_abstractmethoderror_valid_methodtype
    def test_abstractmethoderror_valid_methodtype(methodtype):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 15, in test_abstractmethoderror_valid_methodtype
    error_str = str(error)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/errors/__init__.py", line 305, in __str__
    name = self.class_instance.__name__
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'DummyClass' object has no attribute '__name__'. Did you mean: '__ne__'?
Falsifying example: test_abstractmethoderror_valid_methodtype(
    methodtype='classmethod',
)
```
</details>

## Reproducing the Bug

```python
import pandas as pd

class MyClass:
    pass

instance = MyClass()
error = pd.errors.AbstractMethodError(instance, methodtype='classmethod')
print(str(error))
```

<details>

<summary>
AttributeError when converting AbstractMethodError to string
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/52/repo.py", line 8, in <module>
    print(str(error))
          ~~~^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/errors/__init__.py", line 305, in __str__
    name = self.class_instance.__name__
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'MyClass' object has no attribute '__name__'. Did you mean: '__ne__'?
```
</details>

## Why This Is A Bug

This bug violates a fundamental Python contract: **all exceptions must be convertible to strings without raising an error**. The issue occurs because:

1. **Incorrect Assumption in `__str__` Method**: The `AbstractMethodError.__str__` method at line 305 assumes that when `methodtype='classmethod'`, the `class_instance` parameter is always a class object with a `__name__` attribute. However, instances don't have this attribute.

2. **No Input Validation**: The constructor accepts any object for `class_instance` without type checking, suggesting both classes and instances are valid inputs. The parameter is even ambiguously named `class_instance`, implying it can be either.

3. **Inconsistent Behavior**: All other methodtypes ('method', 'staticmethod', 'property') work correctly with instance objects - they use `type(self.class_instance).__name__` to get the class name. Only 'classmethod' fails.

4. **Documentation Ambiguity**: The documentation never explicitly states that a class object must be passed when `methodtype='classmethod'`. The docstring examples show best practices but don't constitute a requirement.

5. **Exception Display Failure**: This bug can cause secondary failures in error reporting, logging, and debugging tools that expect to be able to safely convert any exception to a string.

## Relevant Context

The bug is located in `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/errors/__init__.py` at lines 303-308:

```python
def __str__(self) -> str:
    if self.methodtype == "classmethod":
        name = self.class_instance.__name__  # Line 305 - problematic line
    else:
        name = type(self.class_instance).__name__
    return f"This {self.methodtype} must be defined in the concrete class {name}"
```

The docstring at line 291 also contains a documentation bug - it incorrectly says "This classmethod must be defined" for an instance method example.

This error would commonly occur when:
- Users are programmatically creating `AbstractMethodError` instances
- Framework code is generating these errors based on user input
- Testing frameworks are exercising error paths

## Proposed Fix

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