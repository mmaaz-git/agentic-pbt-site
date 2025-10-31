# Bug Report: pandas.errors.AbstractMethodError Crashes on __str__ with Invalid classmethod Input

**Target**: `pandas.errors.AbstractMethodError.__str__`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

When `AbstractMethodError` is instantiated with `methodtype="classmethod"` and an instance object instead of a class, calling `str()` on the error raises an `AttributeError` because the code incorrectly assumes `class_instance` will be a class with a `__name__` attribute.

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

# Run the test
test_abstract_method_error_valid_methodtype()
```

<details>

<summary>
**Failing input**: `valid_methodtype='classmethod'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/24/hypo.py", line 21, in <module>
    test_abstract_method_error_valid_methodtype()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/24/hypo.py", line 10, in test_abstract_method_error_valid_methodtype
    def test_abstract_method_error_valid_methodtype(valid_methodtype):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/24/hypo.py", line 16, in test_abstract_method_error_valid_methodtype
    error_message = str(error)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/errors/__init__.py", line 305, in __str__
    name = self.class_instance.__name__
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'DummyClass' object has no attribute '__name__'. Did you mean: '__ne__'?
Falsifying example: test_abstract_method_error_valid_methodtype(
    valid_methodtype='classmethod',
)
```
</details>

## Reproducing the Bug

```python
import pandas.errors as pd_errors


class DummyClass:
    pass


dummy = DummyClass()
error = pd_errors.AbstractMethodError(dummy, methodtype="classmethod")

str(error)
```

<details>

<summary>
AttributeError when converting error to string
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/24/repo.py", line 11, in <module>
    str(error)
    ~~~^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/errors/__init__.py", line 305, in __str__
    name = self.class_instance.__name__
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'DummyClass' object has no attribute '__name__'. Did you mean: '__ne__'?
```
</details>

## Why This Is A Bug

This violates expected behavior in multiple ways:

1. **Inconsistent Input Validation**: The `__init__` method (lines 294-301) accepts any object as `class_instance` without validation, but the `__str__` method (lines 303-308) assumes specific types based on `methodtype`.

2. **Incorrect Assumption in __str__**: At line 305, when `methodtype == "classmethod"`, the code directly accesses `self.class_instance.__name__`, assuming it's a class object. However, instance objects don't have a `__name__` attribute.

3. **Fail-Late Instead of Fail-Fast**: The error occurs during string conversion, not at object creation time when the invalid input is provided. This makes debugging harder for users.

4. **Misleading Parameter Name**: The parameter `class_instance` is ambiguous - it accepts both classes and instances, but the name doesn't clearly communicate this dual purpose.

5. **Documentation vs Implementation Mismatch**: While the docstring examples (lines 279-291) show that classmethods should pass `cls` (the class itself), this requirement is neither explicitly documented nor enforced by the code.

## Relevant Context

The docstring at lines 279-291 in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/errors/__init__.py` shows the intended usage:

```python
class Foo:
    @classmethod
    def classmethod(cls):
        raise pd.errors.AbstractMethodError(cls, methodtype="classmethod")
    def method(self):
        raise pd.errors.AbstractMethodError(self)
```

This clearly indicates that classmethods should pass the class (`cls`), not an instance (`self`). However, the code doesn't validate this expectation.

The bug manifests because:
- For `methodtype="method"`, `"staticmethod"`, and `"property"`: The code uses `type(self.class_instance).__name__` (line 307), which works for both classes and instances
- For `methodtype="classmethod"`: The code uses `self.class_instance.__name__` (line 305), which only works for classes

## Proposed Fix

Make the `__str__` method more robust by handling both classes and instances consistently:

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