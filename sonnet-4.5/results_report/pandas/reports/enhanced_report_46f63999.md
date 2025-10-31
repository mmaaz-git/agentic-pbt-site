# Bug Report: pandas.errors.AbstractMethodError Swapped Variables in Error Message

**Target**: `pandas.errors.AbstractMethodError`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `AbstractMethodError.__init__` method raises a `ValueError` when an invalid `methodtype` is provided, but the error message has swapped variables - showing the invalid input where valid types should be listed and vice versa.

## Property-Based Test

```python
import pytest
from hypothesis import given, strategies as st, assume
import pandas.errors as pe


@given(st.text(min_size=1))
def test_abstractmethoderror_error_message_shows_correct_values(invalid_methodtype):
    """
    Property: When AbstractMethodError raises ValueError for invalid methodtype,
    the error message should correctly display:
    1. The set of valid types in the "must be one of X" part
    2. The invalid value provided in the "got Y instead" part

    This is a basic contract: error messages should accurately describe what
    went wrong by showing valid options and the invalid input.
    """
    valid_types = {"method", "classmethod", "staticmethod", "property"}

    assume(invalid_methodtype not in valid_types)

    class DummyClass:
        pass
    obj = DummyClass()

    with pytest.raises(ValueError) as exc_info:
        pe.AbstractMethodError(obj, methodtype=invalid_methodtype)

    error_message = str(exc_info.value)

    assert "methodtype must be one of" in error_message

    msg_parts = error_message.split("got")
    assert len(msg_parts) == 2

    first_part = msg_parts[0]
    second_part = msg_parts[1]

    for valid_type in valid_types:
        assert valid_type in first_part, \
            f"Valid type '{valid_type}' should appear in first part (before 'got'), but got: {error_message}"

    assert invalid_methodtype in second_part, \
        f"Invalid type '{invalid_methodtype}' should appear in second part (after 'got'), but got: {error_message}"
```

<details>

<summary>
**Failing input**: `'0'`
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0
rootdir: /home/npc/pbt/agentic-pbt/worker_/47
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collected 1 item

hypo.py F                                                                [100%]

=================================== FAILURES ===================================
_________ test_abstractmethoderror_error_message_shows_correct_values __________
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/47/hypo.py", line 7, in test_abstractmethoderror_error_message_shows_correct_values
  |     def test_abstractmethoderror_error_message_shows_correct_values(invalid_methodtype):
  |                    ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/47/hypo.py", line 33, in test_abstractmethoderror_error_message_shows_correct_values
    |     assert len(msg_parts) == 2
    | AssertionError: assert 3 == 2
    |  +  where 3 = len(['methodtype must be one of ', ', ', " {'classmethod', 'method', 'staticmethod', 'property'} instead."])
    | Falsifying example: test_abstractmethoderror_error_message_shows_correct_values(
    |     invalid_methodtype='got',
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/47/hypo.py", line 39, in test_abstractmethoderror_error_message_shows_correct_values
    |     assert valid_type in first_part, \
    |         f"Valid type '{valid_type}' should appear in first part (before 'got'), but got: {error_message}"
    | AssertionError: Valid type 'classmethod' should appear in first part (before 'got'), but got: methodtype must be one of 0, got {'classmethod', 'method', 'staticmethod', 'property'} instead.
    | assert 'classmethod' in 'methodtype must be one of 0, '
    | Falsifying example: test_abstractmethoderror_error_message_shows_correct_values(
    |     invalid_methodtype='0',
    | )
    +------------------------------------
============================ Hypothesis Statistics =============================
hypo.py::test_abstractmethoderror_error_message_shows_correct_values:

  - during reuse phase (0.03 seconds):
    - Typical runtimes: ~ 5-20 ms, of which < 1ms in data generation
    - 0 passing examples, 2 failing examples, 0 invalid examples
    - Found 2 distinct errors in this phase

  - Stopped because nothing left to do


=========================== short test summary info ============================
FAILED hypo.py::test_abstractmethoderror_error_message_shows_correct_values
============================== 1 failed in 0.26s ===============================
```
</details>

## Reproducing the Bug

```python
import pandas.errors as pe


class DummyClass:
    pass


obj = DummyClass()

try:
    pe.AbstractMethodError(obj, methodtype='0')
except ValueError as e:
    print(f"Actual error message:")
    print(f"  {e}")
    print()
    print(f"Expected error message:")
    print(f"  methodtype must be one of {{'method', 'classmethod', 'staticmethod', 'property'}}, got 0 instead.")
```

<details>

<summary>
ValueError with incorrect message formatting
</summary>
```
Actual error message:
  methodtype must be one of 0, got {'staticmethod', 'classmethod', 'property', 'method'} instead.

Expected error message:
  methodtype must be one of {'method', 'classmethod', 'staticmethod', 'property'}, got 0 instead.
```
</details>

## Why This Is A Bug

This violates the universal convention for validation error messages where "must be one of X" lists valid options and "got Y" shows the invalid input provided. The current implementation at line 298 of `pandas/errors/__init__.py` has the f-string variables in the wrong order, creating a confusing error message that tells users "methodtype must be one of 0" when '0' is actually the invalid value they provided. While the ValueError is still raised correctly and both the valid options and invalid input are present in the message, having them in swapped positions contradicts user expectations and standard Python error message patterns, potentially causing confusion when debugging.

## Relevant Context

The bug exists in the pandas errors module at `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/errors/__init__.py:298`. The AbstractMethodError class is used throughout pandas to provide clearer error messages for abstract methods that must be implemented in concrete classes. The validation of the `methodtype` parameter ensures only valid method types ('method', 'classmethod', 'staticmethod', 'property') are accepted. This is a straightforward formatting bug where the variables in the f-string template are simply in the wrong positions.

Documentation link: https://pandas.pydata.org/docs/reference/api/pandas.errors.AbstractMethodError.html

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