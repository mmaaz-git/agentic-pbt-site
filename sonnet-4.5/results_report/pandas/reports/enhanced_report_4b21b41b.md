# Bug Report: pandas.errors.AbstractMethodError Swapped Arguments in Error Message

**Target**: `pandas.errors.AbstractMethodError.__init__`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `AbstractMethodError` class produces a confusing error message when an invalid `methodtype` parameter is provided, with the invalid value and valid types swapped in the message template.

## Property-Based Test

```python
from hypothesis import given, strategies as st, example
import pandas.errors as pd_errors
import pytest


class DummyClass:
    pass


@given(st.text().filter(lambda x: x not in {"method", "classmethod", "staticmethod", "property"} and x != ""))
@example("invalid_type")  # Provide explicit example
def test_abstract_method_error_invalid_methodtype_message(invalid_methodtype):
    dummy = DummyClass()
    with pytest.raises(ValueError) as exc_info:
        pd_errors.AbstractMethodError(dummy, methodtype=invalid_methodtype)

    error_message = str(exc_info.value)
    valid_types = {"method", "classmethod", "staticmethod", "property"}

    # Check that the error message format is correct
    # It should be "methodtype must be one of {valid_types}, got {invalid_methodtype} instead."
    # But the bug causes it to be "methodtype must be one of {invalid_methodtype}, got {valid_types} instead."

    # This assertion will fail due to the bug - the message has the arguments swapped
    expected_pattern = f"methodtype must be one of"
    assert expected_pattern in error_message

    # The bug: it says "must be one of invalid_type" instead of "must be one of {valid_types}"
    if f"one of {invalid_methodtype}" in error_message:
        print(f"BUG DETECTED: Error message incorrectly says 'must be one of {invalid_methodtype}'")
        print(f"Actual error message: {error_message}")
        assert False, f"Bug: error message has swapped arguments - says 'must be one of {invalid_methodtype}' instead of listing valid types"


if __name__ == "__main__":
    # Run the test with hypothesis
    test_abstract_method_error_invalid_methodtype_message()
```

<details>

<summary>
**Failing input**: `invalid_methodtype='invalid_type'`
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/49
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_abstract_method_error_invalid_methodtype_message FAILED    [100%]

=================================== FAILURES ===================================
____________ test_abstract_method_error_invalid_methodtype_message _____________
hypo.py:11: in test_abstract_method_error_invalid_methodtype_message
    @example("invalid_type")  # Provide explicit example
                   ^^^
/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py:1613: in _raise_to_user
    raise the_error_hypothesis_found
hypo.py:32: in test_abstract_method_error_invalid_methodtype_message
    assert False, f"Bug: error message has swapped arguments - says 'must be one of {invalid_methodtype}' instead of listing valid types"
E   AssertionError: Bug: error message has swapped arguments - says 'must be one of invalid_type' instead of listing valid types
E   assert False
E   Falsifying explicit example: test_abstract_method_error_invalid_methodtype_message(
E       invalid_methodtype='invalid_type',
E   )
----------------------------- Captured stdout call -----------------------------
BUG DETECTED: Error message incorrectly says 'must be one of invalid_type'
Actual error message: methodtype must be one of invalid_type, got {'staticmethod', 'property', 'classmethod', 'method'} instead.
=========================== short test summary info ============================
FAILED hypo.py::test_abstract_method_error_invalid_methodtype_message - Asser...
============================== 1 failed in 0.36s ===============================
```
</details>

## Reproducing the Bug

```python
import pandas.errors as pd_errors


class DummyClass:
    pass


dummy = DummyClass()

try:
    pd_errors.AbstractMethodError(dummy, methodtype="invalid_type")
except ValueError as e:
    print(f"Error message: {e}")
```

<details>

<summary>
ValueError raised with confusing error message
</summary>
```
Error message: methodtype must be one of invalid_type, got {'classmethod', 'staticmethod', 'property', 'method'} instead.
```
</details>

## Why This Is A Bug

The error message is meant to inform users which valid method types are acceptable when they provide an invalid one. However, due to a formatting error in line 298 of `/pandas/errors/__init__.py`, the message displays the invalid input where the valid options should appear, and vice versa.

The current message states: "methodtype must be one of invalid_type, got {'property', 'staticmethod', 'method', 'classmethod'} instead."

This is backwards - it's telling the user they must use the invalid value they just provided, while showing the actual valid options as if they were the problematic input. The correct message should be: "methodtype must be one of {'property', 'staticmethod', 'method', 'classmethod'}, got invalid_type instead."

This violates the principle of helpful error messages and could confuse users trying to understand what valid values they should provide.

## Relevant Context

The `AbstractMethodError` class is used internally by pandas to indicate when abstract methods must be implemented by concrete subclasses. The `methodtype` parameter accepts one of four string values: "method", "classmethod", "staticmethod", or "property".

The bug is located in the `__init__` method at line 298 of the pandas errors module:
- Source code location: `/pandas/errors/__init__.py:298`
- The f-string format has the placeholders reversed

This is a simple typo that has likely existed unnoticed because:
1. This error path is rarely triggered in normal usage
2. Even with the swapped message, users can often figure out the correct values from context
3. The error is still raised correctly; only the message is wrong

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