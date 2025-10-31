# Bug Report: pandas.errors.AbstractMethodError Swapped Error Message Variables

**Target**: `pandas.errors.AbstractMethodError.__init__`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `AbstractMethodError.__init__` method has swapped variables in its ValueError format string, causing it to display the invalid input where it should show valid options, and vice versa.

## Property-Based Test

```python
import pandas.errors
from hypothesis import given, strategies as st
import pytest


@given(st.text(min_size=1).filter(lambda x: x not in {"method", "classmethod", "staticmethod", "property"}))
def test_abstract_method_error_message_format(invalid_methodtype):
    with pytest.raises(ValueError) as exc_info:
        pandas.errors.AbstractMethodError(object(), methodtype=invalid_methodtype)

    error_message = str(exc_info.value)

    valid_types_set = {"method", "classmethod", "staticmethod", "property"}

    for valid_type in valid_types_set:
        assert valid_type in error_message, (
            f"Valid type '{valid_type}' should appear in error message, "
            f"but error message is: '{error_message}'"
        )

    parts_after_got = error_message.split("got")
    assert len(parts_after_got) > 1, "Error message should contain 'got'"

    got_part = parts_after_got[1]
    assert invalid_methodtype in got_part, (
        f"The invalid input '{invalid_methodtype}' should appear after 'got' "
        f"in the error message, but the part after 'got' is: '{got_part}'"
    )

# Run the test
if __name__ == "__main__":
    test_abstract_method_error_message_format()
```

<details>

<summary>
**Failing input**: `'0'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/19/hypo.py", line 32, in <module>
    test_abstract_method_error_message_format()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/19/hypo.py", line 7, in test_abstract_method_error_message_format
    def test_abstract_method_error_message_format(invalid_methodtype):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/19/hypo.py", line 25, in test_abstract_method_error_message_format
    assert invalid_methodtype in got_part, (
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: The invalid input '0' should appear after 'got' in the error message, but the part after 'got' is: ' {'staticmethod', 'method', 'property', 'classmethod'} instead.'
Falsifying example: test_abstract_method_error_message_format(
    invalid_methodtype='0',
)
```
</details>

## Reproducing the Bug

```python
import pandas.errors

# Attempting to create an AbstractMethodError with an invalid methodtype
try:
    pandas.errors.AbstractMethodError(object(), methodtype="invalid")
except ValueError as e:
    print(f"Actual error message: {e}")
    print(f"\nExpected error message: methodtype must be one of {{'method', 'classmethod', 'staticmethod', 'property'}}, got invalid instead.")
```

<details>

<summary>
ValueError demonstrates swapped variables in error message
</summary>
```
Actual error message: methodtype must be one of invalid, got {'method', 'property', 'classmethod', 'staticmethod'} instead.

Expected error message: methodtype must be one of {'method', 'classmethod', 'staticmethod', 'property'}, got invalid instead.
```
</details>

## Why This Is A Bug

This violates expected error message conventions and user expectations in several ways:

1. **Nonsensical Error Message**: The current message says "methodtype must be one of invalid" which is illogical - it's telling the user that the valid value is the invalid input they provided.

2. **Standard Error Message Convention Violation**: Python error messages conventionally follow the pattern "parameter must be one of [valid options], got [invalid input] instead". This error has it backwards.

3. **User Confusion**: When debugging, developers rely on error messages to understand what went wrong. This backwards message will confuse users about what the valid options are versus what they incorrectly provided.

4. **Documentation Inconsistency**: While the pandas documentation doesn't explicitly specify the error message format, standard Python conventions and common sense dictate that error messages should clearly distinguish between valid options and the invalid input received.

## Relevant Context

The bug is located in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/errors/__init__.py` at line 298. The AbstractMethodError class is designed to provide clearer error messages than NotImplementedError for abstract methods in concrete classes.

The methodtype parameter accepts four valid values: 'method', 'classmethod', 'staticmethod', and 'property'. When an invalid value is provided, the ValueError should inform users of these valid options, not tell them that their invalid input is what's required.

This error occurs during the initialization of AbstractMethodError, before the actual AbstractMethodError is raised, so it's a validation error that prevents the intended error from being created.

Link to source code: [pandas/errors/__init__.py:298](https://github.com/pandas-dev/pandas/blob/main/pandas/errors/__init__.py#L298)

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