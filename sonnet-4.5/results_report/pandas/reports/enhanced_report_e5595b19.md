# Bug Report: pandas.errors.AbstractMethodError Swapped Format String Parameters in Validation Message

**Target**: `pandas.errors.AbstractMethodError.__init__`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The ValueError message in `AbstractMethodError.__init__` has its format string parameters reversed, causing the error message to display the user's invalid input where it should show the valid options, and vice versa.

## Property-Based Test

```python
from hypothesis import given, strategies as st, example
import pandas as pd
import pytest


@given(st.text().filter(lambda x: x not in {"method", "classmethod", "staticmethod", "property"}))
@example("invalid_type")  # Explicitly provide the failing example
def test_abstract_method_error_validation_message(invalid_methodtype):
    class DummyClass:
        pass

    instance = DummyClass()

    with pytest.raises(ValueError) as exc_info:
        pd.errors.AbstractMethodError(instance, methodtype=invalid_methodtype)

    error_message = str(exc_info.value)

    valid_types = {"method", "classmethod", "staticmethod", "property"}

    # The bug: The error message has swapped parameters
    # It says "methodtype must be one of invalid_type, got {'method', ...} instead"
    # When it should say "methodtype must be one of {'method', ...}, got invalid_type instead"

    # This assertion will FAIL due to the bug
    # We expect the valid types to appear after "must be one of"
    if "must be one of" in error_message:
        parts = error_message.split("must be one of")[1].split(",")[0].strip()
        # The bug causes parts to be the invalid_methodtype, not the valid types
        print(f"Test failed for input '{invalid_methodtype}'")
        print(f"Error message: {error_message}")
        print(f"After 'must be one of': '{parts}'")
        print(f"Expected to see valid types but saw: '{parts}'")

        # This assertion demonstrates the bug
        assert parts != invalid_methodtype, f"Bug confirmed: Error message has swapped parameters. The invalid input '{invalid_methodtype}' appears where valid options should be."
```

<details>

<summary>
**Failing input**: `"invalid_type"`
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/63
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_abstract_method_error_validation_message Test failed for input 'invalid_type'
Error message: methodtype must be one of invalid_type, got {'staticmethod', 'classmethod', 'property', 'method'} instead.
After 'must be one of': 'invalid_type'
Expected to see valid types but saw: 'invalid_type'
FAILED

=================================== FAILURES ===================================
________________ test_abstract_method_error_validation_message _________________

    @given(st.text().filter(lambda x: x not in {"method", "classmethod", "staticmethod", "property"}))
>   @example("invalid_type")  # Explicitly provide the failing example
                   ^^^

hypo.py:7:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py:1613: in _raise_to_user
    raise the_error_hypothesis_found
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

invalid_methodtype = 'invalid_type'

    @given(st.text().filter(lambda x: x not in {"method", "classmethod", "staticmethod", "property"}))
    @example("invalid_type")  # Explicitly provide the failing example
    def test_abstract_method_error_validation_message(invalid_methodtype):
        class DummyClass:
            pass

        instance = DummyClass()

        with pytest.raises(ValueError) as exc_info:
            pd.errors.AbstractMethodError(instance, methodtype=invalid_methodtype)

        error_message = str(exc_info.value)

        valid_types = {"method", "classmethod", "staticmethod", "property"}

        # The bug: The error message has swapped parameters
        # It says "methodtype must be one of invalid_type, got {'method', ...} instead"
        # When it should say "methodtype must be one of {'method', ...}, got invalid_type instead"

        # This assertion will FAIL due to the bug
        # We expect the valid types to appear after "must be one of"
        if "must be one of" in error_message:
            parts = error_message.split("must be one of")[1].split(",")[0].strip()
            # The bug causes parts to be the invalid_methodtype, not the valid types
            print(f"Test failed for input '{invalid_methodtype}'")
            print(f"Error message: {error_message}")
            print(f"After 'must be one of': '{parts}'")
            print(f"Expected to see valid types but saw: '{parts}'")

            # This assertion demonstrates the bug
>           assert parts != invalid_methodtype, f"Bug confirmed: Error message has swapped parameters. The invalid input '{invalid_methodtype}' appears where valid options should be."
E           AssertionError: Bug confirmed: Error message has swapped parameters. The invalid input 'invalid_type' appears where valid options should be.
E           assert 'invalid_type' != 'invalid_type'
E           Falsifying explicit example: test_abstract_method_error_validation_message(
E               invalid_methodtype='invalid_type',
E           )

hypo.py:36: AssertionError
=========================== short test summary info ============================
FAILED hypo.py::test_abstract_method_error_validation_message - AssertionErro...
============================== 1 failed in 0.35s ===============================
```
</details>

## Reproducing the Bug

```python
import pandas as pd


class DummyClass:
    pass


instance = DummyClass()

try:
    pd.errors.AbstractMethodError(instance, methodtype="invalid_type")
except ValueError as e:
    print(f"Error message: {e}")
```

<details>

<summary>
ValueError with incorrect parameter order in message
</summary>
```
Error message: methodtype must be one of invalid_type, got {'method', 'classmethod', 'staticmethod', 'property'} instead.
```
</details>

## Why This Is A Bug

This violates the universal convention for error messages in Python and programming in general. When an error message uses the pattern "must be one of X, got Y instead", developers universally expect:
- X to be the set of valid options
- Y to be their invalid input

The current implementation has these reversed due to incorrect variable placement in the f-string on line 298 of `pandas/errors/__init__.py`. The code currently reads:
```python
f"methodtype must be one of {methodtype}, got {types} instead."
```

Where:
- `methodtype` contains the user's invalid input (e.g., "invalid_type")
- `types` contains the valid options set `{"method", "classmethod", "staticmethod", "property"}`

This creates a confusing error message that tells users their invalid input is what's expected, and the valid options are what they incorrectly provided. While the validation logic itself works correctly (invalid values are properly rejected), the misleading error message significantly impacts developer experience during debugging.

## Relevant Context

The `AbstractMethodError` class is used throughout pandas to provide clear error messages when abstract methods are not implemented in concrete classes. The constructor accepts an optional `methodtype` parameter that must be one of four valid strings: "method", "classmethod", "staticmethod", or "property".

When an invalid `methodtype` is provided, the constructor raises a `ValueError` before the `AbstractMethodError` is created. This validation is important to ensure consistent error messages throughout the pandas codebase.

The bug is located in the pandas source at:
- File: `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/errors/__init__.py`
- Line: 298
- Function: `AbstractMethodError.__init__`

The pandas documentation for `AbstractMethodError` shows example usage but doesn't explicitly document the format of the validation error message. However, following Python's established conventions for error messages is essential for maintaining a consistent and intuitive developer experience.

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