# Bug Report: pandas.errors.AbstractMethodError Swapped Error Message Variables

**Target**: `pandas.errors.AbstractMethodError`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The AbstractMethodError.__init__ method produces a backwards error message when invalid methodtype values are provided, showing the invalid input as the expected values and the valid options as what was provided.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas.errors
import pytest


class DummyClass:
    pass


@given(st.text(min_size=1).filter(lambda x: x not in {"method", "classmethod", "staticmethod", "property"}))
@settings(max_examples=100)
def test_abstractmethoderror_invalid_methodtype_message(invalid_methodtype):
    valid_types = {"method", "classmethod", "staticmethod", "property"}

    with pytest.raises(ValueError) as exc_info:
        pandas.errors.AbstractMethodError(DummyClass(), methodtype=invalid_methodtype)

    error_message = str(exc_info.value)

    for valid_type in valid_types:
        assert valid_type in error_message

    assert invalid_methodtype in error_message

    msg_start = error_message.split(',')[0]
    assert invalid_methodtype not in msg_start


if __name__ == "__main__":
    test_abstractmethoderror_invalid_methodtype_message()
```

<details>

<summary>
**Failing input**: `invalid_methodtype='0'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 30, in <module>
    test_abstractmethoderror_invalid_methodtype_message()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 11, in test_abstractmethoderror_invalid_methodtype_message
    @settings(max_examples=100)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 26, in test_abstractmethoderror_invalid_methodtype_message
    assert invalid_methodtype not in msg_start
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_abstractmethoderror_invalid_methodtype_message(
    invalid_methodtype='0',
)
```
</details>

## Reproducing the Bug

```python
import pandas as pd


class Foo:
    pass


try:
    pd.errors.AbstractMethodError(Foo(), methodtype="invalid_type")
except ValueError as e:
    print(str(e))
```

<details>

<summary>
ValueError with backwards error message
</summary>
```
methodtype must be one of invalid_type, got {'classmethod', 'staticmethod', 'method', 'property'} instead.
```
</details>

## Why This Is A Bug

This violates expected behavior because the error message is logically backwards. When a ValueError is raised for an invalid parameter, the error message should clearly communicate:
1. What values are acceptable
2. What invalid value was actually provided

The current implementation states "methodtype must be one of invalid_type" which incorrectly suggests that 'invalid_type' is the acceptable value, when it's actually the problematic input. The message then shows the set of valid values ({'method', 'classmethod', 'staticmethod', 'property'}) as if they were the invalid input that was provided.

This contradicts standard error message patterns throughout Python and pandas where the format is typically "parameter must be one of [valid options], got [invalid input] instead." The swapped variables make debugging confusing for users who encounter this error.

## Relevant Context

The bug is located in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/errors/__init__.py` at line 298. The AbstractMethodError class is designed to provide better error messages than plain NotImplementedError when abstract methods must be implemented in concrete classes.

The valid methodtype values are: 'method', 'classmethod', 'staticmethod', and 'property'. These are checked against a hardcoded set in the __init__ method. When an invalid value is provided, the validation correctly raises a ValueError, but with the incorrect message formatting.

While this error only appears when users incorrectly instantiate AbstractMethodError (which shouldn't happen in normal usage as it's typically raised by library code), clear error messages are important for debugging and understanding what went wrong.

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