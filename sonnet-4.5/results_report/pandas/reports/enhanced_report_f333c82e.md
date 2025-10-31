# Bug Report: pandas.errors.AbstractMethodError Crashes on str() When Instance Passed for Classmethod

**Target**: `pandas.errors.AbstractMethodError`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `AbstractMethodError.__str__` method crashes with an `AttributeError` when `methodtype="classmethod"` and an instance object is passed instead of a class, resulting in an unhelpful error message instead of proper validation.

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


if __name__ == "__main__":
    test_abstractmethoderror_str_crashes_with_instance_for_classmethod()
```

<details>

<summary>
**Failing input**: `methodtype="classmethod"` with any instance object
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/32
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_abstractmethoderror_str_crashes_with_instance_for_classmethod PASSED [100%]

============================== 1 passed in 1.55s ===============================
```
</details>

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

<details>

<summary>
Crashes with AttributeError when str() is called
</summary>
```
Crashed with AttributeError: 'SampleClass' object has no attribute '__name__'
```
</details>

## Why This Is A Bug

This violates expected error handling behavior. When users make a mistake by passing an instance instead of a class for `methodtype="classmethod"`, they receive a cryptic `AttributeError` about a missing `__name__` attribute rather than a clear error message explaining what went wrong.

The documentation examples in the docstring (lines 279-291 of `/home/npc/miniconda/lib/python3.13/site-packages/pandas/errors/__init__.py`) show that when `methodtype="classmethod"`, the first parameter should be `cls` (the class itself). However, the code doesn't validate this assumption, and the parameter name `class_instance` is ambiguous as it could mean either a class or an instance.

The bug occurs in the `__str__` method at line 305, where the code assumes `class_instance` has a `__name__` attribute (which only classes have) when `methodtype="classmethod"`, but instances don't have this attribute. The else branch at line 307 correctly handles instances by using `type(self.class_instance).__name__`, showing that the code already knows how to handle both cases but doesn't apply this consistently.

## Relevant Context

The AbstractMethodError class is used to provide clear error messages when abstract methods are not implemented in concrete classes. The docstring examples demonstrate correct usage:
- For classmethods: `raise pd.errors.AbstractMethodError(cls, methodtype="classmethod")`
- For regular methods: `raise pd.errors.AbstractMethodError(self)`

The parameter `class_instance` accepts both classes and instances, but when `methodtype="classmethod"`, it expects a class object. The code makes this assumption in line 305 but doesn't validate it during initialization or handle it gracefully in `__str__`.

Documentation link: The error class is defined in pandas/errors/__init__.py lines 273-308.

## Proposed Fix

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