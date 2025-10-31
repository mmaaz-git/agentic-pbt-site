# Bug Report: pandas.errors.AbstractMethodError.__str__ Crashes When Instance Passed for Classmethod

**Target**: `pandas.errors.AbstractMethodError.__str__`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `AbstractMethodError.__str__` method crashes with `AttributeError` when the error is created with `methodtype="classmethod"` but an instance is passed instead of a class, because it tries to access `__name__` attribute directly on the instance.

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


if __name__ == "__main__":
    test_abstract_method_error_valid_methodtype()
```

<details>

<summary>
**Failing input**: `valid_type='classmethod'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/11/hypo.py", line 21, in <module>
    test_abstract_method_error_valid_methodtype()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/11/hypo.py", line 10, in test_abstract_method_error_valid_methodtype
    def test_abstract_method_error_valid_methodtype(valid_type):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/11/hypo.py", line 14, in test_abstract_method_error_valid_methodtype
    msg = str(err)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/errors/__init__.py", line 305, in __str__
    name = self.class_instance.__name__
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'DummyClass' object has no attribute '__name__'. Did you mean: '__ne__'?
Falsifying example: test_abstract_method_error_valid_methodtype(
    valid_type='classmethod',
)
```
</details>

## Reproducing the Bug

```python
import pandas.errors


class DummyClass:
    pass


# Create an instance (not a class)
instance = DummyClass()

# Create AbstractMethodError with methodtype="classmethod" but passing an instance
err = pandas.errors.AbstractMethodError(instance, methodtype="classmethod")

# Try to convert to string - this should crash
try:
    msg = str(err)
    print(f"String representation: {msg}")
except AttributeError as e:
    print(f"AttributeError: {e}")
```

<details>

<summary>
AttributeError when converting to string
</summary>
```
AttributeError: 'DummyClass' object has no attribute '__name__'
```
</details>

## Why This Is A Bug

This violates the fundamental contract that objects that can be successfully created should be usable. The `__init__` method accepts any `class_instance` parameter without validating it matches the `methodtype`, but the `__str__` method makes incompatible assumptions:

1. When `methodtype="classmethod"`, `__str__` assumes `class_instance` is a class and directly accesses `class_instance.__name__` (line 305)
2. For all other methodtypes, `__str__` assumes it's an instance and uses `type(class_instance).__name__` (line 307)

The documentation examples show passing `cls` for classmethods and `self` for instance methods, but this requirement is never explicitly stated or enforced. The parameter name `class_instance` is ambiguous - it could mean either "class or instance". There's also a documentation bug on line 291 where the instance method example incorrectly says "This classmethod" instead of "This method".

Additionally, there's an inconsistency with `methodtype="staticmethod"` - logically it should expect a class like classmethod, but it falls into the else branch and accidentally works with instances.

## Relevant Context

The bug is located in `/pandas/errors/__init__.py` at lines 273-308. The issue specifically occurs at line 305 where the code tries to access `self.class_instance.__name__` when `methodtype == "classmethod"`.

The valid methodtypes are: `{"method", "classmethod", "staticmethod", "property"}`, validated in `__init__`.

Documentation examples from the source:
- For classmethods: `raise pd.errors.AbstractMethodError(cls, methodtype="classmethod")`
- For instance methods: `raise pd.errors.AbstractMethodError(self)`

The pandas API documentation doesn't explicitly specify that `class_instance` must be a class when using `methodtype="classmethod"`.

## Proposed Fix

Add validation in `__init__` to ensure the correct type is passed for each methodtype:

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