# Bug Report: pandas.errors.AbstractMethodError Crashes When Converting to String with methodtype='classmethod'

**Target**: `pandas.errors.AbstractMethodError.__str__`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`AbstractMethodError.__str__` crashes with `AttributeError` when `methodtype='classmethod'` and `class_instance` is an instance object rather than a class, because it tries to access the `__name__` attribute which doesn't exist on instances.

## Property-Based Test

```python
from hypothesis import given
import hypothesis.strategies as st
import pandas.errors as pd_errors


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


if __name__ == "__main__":
    test_abstract_method_error_valid_methodtype()
```

<details>

<summary>
**Failing input**: `valid_methodtype='classmethod'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 24, in <module>
    test_abstract_method_error_valid_methodtype()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 7, in test_abstract_method_error_valid_methodtype
    def test_abstract_method_error_valid_methodtype(valid_methodtype):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 17, in test_abstract_method_error_valid_methodtype
    error_str = str(error)
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


instance = DummyClass()

error = pd_errors.AbstractMethodError(instance, methodtype="classmethod")

print("Error created successfully")

try:
    error_str = str(error)
    print(f"Error string: {error_str}")
except AttributeError as e:
    print(f"CRASH: {e}")
```

<details>

<summary>
CRASH: AttributeError when converting error to string
</summary>
```
Error created successfully
CRASH: 'DummyClass' object has no attribute '__name__'
```
</details>

## Why This Is A Bug

This bug violates the principle of API consistency. The `__init__` method of `AbstractMethodError` accepts any object as the `class_instance` parameter regardless of the `methodtype` value, but the `__str__` method makes different assumptions about the type of `class_instance` based on `methodtype`.

Specifically, in the `__str__` method (line 303-308 in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/errors/__init__.py`):

```python
def __str__(self) -> str:
    if self.methodtype == "classmethod":
        name = self.class_instance.__name__  # Assumes class_instance is a class
    else:
        name = type(self.class_instance).__name__  # Works for any object
    return f"This {self.methodtype} must be defined in the concrete class {name}"
```

When `methodtype == "classmethod"`, the code assumes `class_instance` has a `__name__` attribute, which is only true for class objects, not instance objects. This causes an `AttributeError` when an instance is passed.

The documentation's example (lines 279-287) shows the intended usage where `cls` is passed for classmethods, but there's no validation in `__init__` to enforce this requirement, nor does the parameter name `class_instance` clearly indicate it should be a class for classmethods.

## Relevant Context

The pandas documentation example shows the correct usage pattern:
- For classmethods: `raise pd.errors.AbstractMethodError(cls, methodtype="classmethod")`
- For regular methods: `raise pd.errors.AbstractMethodError(self)`

However, the API design issues are:
1. The parameter is named `class_instance` which is ambiguous - it suggests both classes and instances are acceptable
2. No validation occurs at initialization time to ensure correct usage
3. The error occurs later during string conversion, making debugging harder

This issue was discovered through property-based testing, which systematically tested all valid `methodtype` values with the same input pattern.

## Proposed Fix

Add validation in the `__init__` method to ensure that when `methodtype='classmethod'`, the `class_instance` parameter is actually a class object:

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