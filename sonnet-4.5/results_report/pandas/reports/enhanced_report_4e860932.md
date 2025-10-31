# Bug Report: pandas.errors.AbstractMethodError Swapped Error Message Parameters

**Target**: `pandas.errors.AbstractMethodError.__init__`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The AbstractMethodError class raises a ValueError with swapped format string parameters when an invalid methodtype is provided, resulting in an error message that displays the invalid value where valid options should be shown and vice versa.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas.errors as errors
import pytest


class DummyClass:
    pass


@given(st.text().filter(lambda x: x not in {"method", "classmethod", "staticmethod", "property"}))
def test_abstract_method_error_validation_message_format(invalid_methodtype):
    with pytest.raises(ValueError) as exc_info:
        errors.AbstractMethodError(DummyClass, methodtype=invalid_methodtype)

    error_msg = str(exc_info.value)

    valid_types = {"method", "classmethod", "staticmethod", "property"}

    assert f"methodtype must be one of {invalid_methodtype}" not in error_msg, \
        f"Bug: Error message incorrectly says 'methodtype must be one of {invalid_methodtype}'"


if __name__ == "__main__":
    test_abstract_method_error_validation_message_format()
```

<details>

<summary>
**Failing input**: `invalid_methodtype=''`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/11/hypo.py", line 24, in <module>
    test_abstract_method_error_validation_message_format()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/11/hypo.py", line 11, in test_abstract_method_error_validation_message_format
    def test_abstract_method_error_validation_message_format(invalid_methodtype):
                  ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/11/hypo.py", line 19, in test_abstract_method_error_validation_message_format
    assert f"methodtype must be one of {invalid_methodtype}" not in error_msg, \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Bug: Error message incorrectly says 'methodtype must be one of '
Falsifying example: test_abstract_method_error_validation_message_format(
    invalid_methodtype='',  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import pandas.errors as errors


class TestClass:
    pass


try:
    errors.AbstractMethodError(TestClass, methodtype="invalid_type")
except ValueError as e:
    print(f"Error message: {e}")
```

<details>

<summary>
ValueError with backwards error message
</summary>
```
Error message: methodtype must be one of invalid_type, got {'property', 'staticmethod', 'method', 'classmethod'} instead.
```
</details>

## Why This Is A Bug

This violates the expected behavior of helpful error messages. The error message format string at line 298 in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/errors/__init__.py` has its parameters reversed. The message currently says "methodtype must be one of [user's invalid value], got [valid options] instead", which is logically backwards. Users expect error messages to show valid options first, then the invalid value they provided. This confusion wastes developer time when debugging, as they must mentally reverse the message to understand what valid values are acceptable. While the validation behavior itself is undocumented, once the code decides to validate and provide an error message, that message should follow standard conventions and make logical sense.

## Relevant Context

The bug is located at line 298 of `pandas/errors/__init__.py`. The AbstractMethodError class is intended to provide clearer error messages for abstract methods that must be implemented in concrete classes. The validation code checks if the methodtype parameter is one of {"method", "classmethod", "staticmethod", "property"}, but the error message format string has the variables in the wrong order.

The pandas documentation does not specify what happens when invalid methodtype values are provided, nor does it document the complete set of valid values. However, the implementation includes validation logic that should provide helpful error messages when invalid values are used.

Source code location: https://github.com/pandas-dev/pandas/blob/main/pandas/errors/__init__.py#L298

## Proposed Fix

```diff
--- a/pandas/errors/__init__.py
+++ b/pandas/errors/__init__.py
@@ -295,7 +295,7 @@ class AbstractMethodError(NotImplementedError):
         types = {"method", "classmethod", "staticmethod", "property"}
         if methodtype not in types:
             raise ValueError(
-                f"methodtype must be one of {methodtype}, got {types} instead."
+                f"methodtype must be one of {types}, got {methodtype} instead."
             )
         self.methodtype = methodtype
         self.class_instance = class_instance
```