# Bug Report: pandas.errors.AbstractMethodError Swapped Error Message Variables

**Target**: `pandas.errors.AbstractMethodError.__init__`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `AbstractMethodError.__init__` method has swapped variables in its error message when an invalid `methodtype` parameter is provided, displaying "methodtype must be one of {invalid_value}, got {valid_types} instead" rather than the correct "methodtype must be one of {valid_types}, got {invalid_value} instead".

## Property-Based Test

```python
import pytest
from hypothesis import given, strategies as st
import pandas.errors as pd_errors


@given(st.text(min_size=1).filter(lambda x: x not in {"method", "classmethod", "staticmethod", "property"}))
def test_abstract_method_error_swapped_values_bug(invalid_methodtype):
    """
    BUG: AbstractMethodError has swapped values in its error message.

    The error message currently says:
        "methodtype must be one of <invalid_value>, got <valid_types> instead."

    But it should say:
        "methodtype must be one of <valid_types>, got <invalid_value> instead."
    """
    class DummyClass:
        pass

    instance = DummyClass()

    with pytest.raises(ValueError) as exc_info:
        pd_errors.AbstractMethodError(instance, methodtype=invalid_methodtype)

    error_msg = str(exc_info.value)

    if "must be one of" in error_msg and "got" in error_msg:
        parts = error_msg.split("got")
        first_part = parts[0]
        second_part = parts[1] if len(parts) > 1 else ""

        valid_types_str = "{'method', 'classmethod', 'staticmethod', 'property'}"

        assert valid_types_str in first_part or all(t in first_part for t in ["method", "classmethod"]), \
            f"Valid types should appear in 'must be one of' clause, but got: {error_msg}"

        assert invalid_methodtype in second_part, \
            f"Invalid value should appear after 'got', but got: {error_msg}"


if __name__ == "__main__":
    test_abstract_method_error_swapped_values_bug()
```

<details>

<summary>
**Failing input**: `invalid_methodtype='0'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 42, in <module>
    test_abstract_method_error_swapped_values_bug()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 7, in test_abstract_method_error_swapped_values_bug
    def test_abstract_method_error_swapped_values_bug(invalid_methodtype):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 34, in test_abstract_method_error_swapped_values_bug
    assert valid_types_str in first_part or all(t in first_part for t in ["method", "classmethod"]), \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Valid types should appear in 'must be one of' clause, but got: methodtype must be one of 0, got {'classmethod', 'method', 'property', 'staticmethod'} instead.
Falsifying example: test_abstract_method_error_swapped_values_bug(
    invalid_methodtype='0',
)
```
</details>

## Reproducing the Bug

```python
import pandas.errors as pd_errors


class DummyClass:
    pass


instance = DummyClass()

try:
    pd_errors.AbstractMethodError(instance, methodtype="invalid")
except ValueError as e:
    print(f"Actual:   {e}")
    print("Expected: methodtype must be one of {'method', 'classmethod', 'staticmethod', 'property'}, got invalid instead.")
```

<details>

<summary>
Error message has swapped values
</summary>
```
Actual:   methodtype must be one of invalid, got {'classmethod', 'staticmethod', 'method', 'property'} instead.
Expected: methodtype must be one of {'method', 'classmethod', 'staticmethod', 'property'}, got invalid instead.
```
</details>

## Why This Is A Bug

This violates the standard Python error message convention where error messages follow the pattern "expected X, got Y" not "expected Y, got X". The current implementation creates a nonsensical error message that says "methodtype must be one of invalid" which is confusing for users trying to debug their code. While the ValueError is still raised correctly, the swapped values make it harder to understand what went wrong and what the valid options actually are.

## Relevant Context

The bug is located in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/errors/__init__.py` at line 298. The AbstractMethodError class is designed to be a more descriptive replacement for NotImplementedError in abstract methods. The methodtype parameter accepts one of four valid values: 'method', 'classmethod', 'staticmethod', or 'property'. When an invalid value is provided, the constructor raises a ValueError with an informative message, but the f-string template has the variables in the wrong positions.

This appears to be a simple typo where `{methodtype}` and `{types}` were accidentally swapped in the f-string. The error only affects the error message text and doesn't impact the actual functionality of the class or the ValueError being raised.

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