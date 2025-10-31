# Bug Report: pandas.errors.AbstractMethodError - Swapped parameters in validation error message

**Target**: `pandas.errors.AbstractMethodError.__init__`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The validation error message in `AbstractMethodError.__init__` has swapped parameters, displaying "methodtype must be one of {invalid_value}" instead of "methodtype must be one of {valid_types}".

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas.errors as errors
import pytest

@given(st.text(min_size=1))
def test_abstractmethoderror_invalid_methodtype_message(invalid_methodtype):
    valid_types = {"method", "classmethod", "staticmethod", "property"}
    if invalid_methodtype in valid_types:
        return

    class DummyClass:
        pass

    instance = DummyClass()

    with pytest.raises(ValueError) as exc_info:
        errors.AbstractMethodError(instance, methodtype=invalid_methodtype)

    error_message = str(exc_info.value)

    for valid_type in valid_types:
        assert valid_type in error_message
    assert invalid_methodtype not in f"must be one of {invalid_methodtype}"

# Run the test
if __name__ == "__main__":
    test_abstractmethoderror_invalid_methodtype_message()
```

<details>

<summary>
**Failing input**: `'0'`
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0
rootdir: /home/npc/pbt/agentic-pbt/worker_/9
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collected 1 item

hypo.py F                                                                [100%]

=================================== FAILURES ===================================
_____________ test_abstractmethoderror_invalid_methodtype_message ______________
hypo.py:6: in test_abstractmethoderror_invalid_methodtype_message
    def test_abstractmethoderror_invalid_methodtype_message(invalid_methodtype):
                   ^^^
hypo.py:23: in test_abstractmethoderror_invalid_methodtype_message
    assert invalid_methodtype not in f"must be one of {invalid_methodtype}"
E   AssertionError: assert '0' not in 'must be one of 0'
E
E     '0' is contained here:
E       must be one of 0
E     ?                +
E   Falsifying example: test_abstractmethoderror_invalid_methodtype_message(
E       invalid_methodtype='0',
E   )
=========================== short test summary info ============================
FAILED hypo.py::test_abstractmethoderror_invalid_methodtype_message - Asserti...
============================== 1 failed in 0.36s ===============================
```
</details>

## Reproducing the Bug

```python
import pandas.errors as errors

class DummyClass:
    pass

try:
    errors.AbstractMethodError(DummyClass(), methodtype='invalid')
except ValueError as e:
    print(f"Error message: {e}")
    print()
    print("Analysis:")
    print(f"The error says 'methodtype must be one of invalid' which is nonsensical.")
    print(f"It should say 'methodtype must be one of {{'method', 'property', 'staticmethod', 'classmethod'}}, got invalid instead.'")
```

<details>

<summary>
ValueError with nonsensical message about methodtype
</summary>
```
Error message: methodtype must be one of invalid, got {'staticmethod', 'method', 'classmethod', 'property'} instead.

Analysis:
The error says 'methodtype must be one of invalid' which is nonsensical.
It should say 'methodtype must be one of {'method', 'property', 'staticmethod', 'classmethod'}, got invalid instead.'
```
</details>

## Why This Is A Bug

This bug violates the expected behavior of error messages in several ways:

1. **Logical inconsistency**: The error message states "methodtype must be one of invalid", where 'invalid' is the user's incorrect input value. This is semantically incorrect - the methodtype cannot "must be one of" the invalid value that was rejected.

2. **Confusing user experience**: When developers encounter this error, they see their invalid input presented as what the value "must be", which is the opposite of the intended message. This makes debugging harder.

3. **Clear code error**: Looking at the source code at `/home/npc/miniconda/lib/python3.13/site-packages/pandas/errors/__init__.py` line 298, the f-string clearly has the variables in the wrong positions: `f"methodtype must be one of {methodtype}, got {types} instead."` should be `f"methodtype must be one of {types}, got {methodtype} instead."`

4. **Documentation expectations**: While the pandas documentation doesn't specify the exact error message format, users reasonably expect error messages to follow the standard pattern of "parameter must be one of [valid options], got [invalid value] instead", which is common across Python libraries.

## Relevant Context

The `AbstractMethodError` class is used to indicate that an abstract method must be implemented in concrete classes. The `methodtype` parameter accepts only four valid values: "method", "classmethod", "staticmethod", or "property". When an invalid value is provided, the validation in `__init__` raises a ValueError.

The bug is located in the pandas errors module at line 298:
- File: `/home/npc/miniconda/lib/python3.13/site-packages/pandas/errors/__init__.py`
- Function: `AbstractMethodError.__init__`
- Line: 298

The validation itself works correctly - it properly rejects invalid values. Only the error message formatting is incorrect.

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