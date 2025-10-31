# Bug Report: pandas.errors.AbstractMethodError Swapped Error Message Arguments

**Target**: `pandas.errors.AbstractMethodError.__init__`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The AbstractMethodError class produces an incorrectly formatted error message when an invalid methodtype is provided, with the invalid value and valid options swapped in the error string.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas.errors as errors
import pytest


@given(
    invalid_methodtype=st.text(min_size=1).filter(
        lambda x: x not in {"method", "classmethod", "staticmethod", "property"}
    )
)
def test_abstractmethoderror_correct_error_format(invalid_methodtype):
    class DummyClass:
        pass

    instance = DummyClass()

    with pytest.raises(ValueError) as exc_info:
        errors.AbstractMethodError(instance, methodtype=invalid_methodtype)

    error_message = str(exc_info.value)

    parts = error_message.split(",", 1)
    first_part = parts[0] if len(parts) > 0 else ""

    assert invalid_methodtype not in first_part, (
        f"Bug: The invalid methodtype '{invalid_methodtype}' should not be in "
        f"'methodtype must be one of X' part. Got: {error_message}"
    )


if __name__ == "__main__":
    # Run the test with Hypothesis
    test_abstractmethoderror_correct_error_format()
```

<details>

<summary>
**Failing input**: `invalid_methodtype='0'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/26/hypo.py", line 33, in <module>
    test_abstractmethoderror_correct_error_format()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/26/hypo.py", line 7, in test_abstractmethoderror_correct_error_format
    invalid_methodtype=st.text(min_size=1).filter(
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/26/hypo.py", line 25, in test_abstractmethoderror_correct_error_format
    assert invalid_methodtype not in first_part, (
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Bug: The invalid methodtype '0' should not be in 'methodtype must be one of X' part. Got: methodtype must be one of 0, got {'staticmethod', 'classmethod', 'method', 'property'} instead.
Falsifying example: test_abstractmethoderror_correct_error_format(
    invalid_methodtype='0',
)
```
</details>

## Reproducing the Bug

```python
import pandas.errors as errors


class DummyClass:
    pass


instance = DummyClass()

try:
    errors.AbstractMethodError(instance, methodtype="invalid")
except ValueError as e:
    print(f"Actual error message: {e}")
    print()
    print("Expected error message: methodtype must be one of {'method', 'classmethod', 'staticmethod', 'property'}, got invalid instead.")
```

<details>

<summary>
ValueError raised with incorrect message format
</summary>
```
Actual error message: methodtype must be one of invalid, got {'staticmethod', 'property', 'method', 'classmethod'} instead.

Expected error message: methodtype must be one of {'method', 'classmethod', 'staticmethod', 'property'}, got invalid instead.
```
</details>

## Why This Is A Bug

This bug violates the fundamental contract of error messages, which should clearly communicate what went wrong and how to fix it. The current implementation produces error messages that state "methodtype must be one of [invalid_value]" instead of showing the valid options. This directly contradicts standard error message conventions where the valid options should be presented first, followed by the invalid input that was provided.

The error message is misleading because it suggests that the invalid input (e.g., "invalid" or "0") is actually what the methodtype should be, when in fact these are the values that caused the error. This confusion can significantly slow down debugging, especially for developers unfamiliar with the pandas codebase who rely on clear error messages to understand API requirements.

## Relevant Context

The bug is located in `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/errors/__init__.py` at line 298. The AbstractMethodError class is designed to provide clearer error messages than the standard NotImplementedError for abstract methods that must be implemented in concrete classes.

The methodtype parameter accepts only four valid values: "method", "classmethod", "staticmethod", or "property". When any other value is provided, a ValueError should be raised with a clear message indicating the valid options.

Documentation reference: The pandas.errors.AbstractMethodError is part of the pandas exceptions module, used throughout the pandas codebase to enforce implementation of abstract methods in subclasses.

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