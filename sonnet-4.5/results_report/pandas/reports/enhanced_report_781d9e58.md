# Bug Report: pandas.errors.AbstractMethodError Error Message Variables Swapped

**Target**: `pandas.errors.AbstractMethodError`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The error message in `AbstractMethodError.__init__` has swapped variables in the f-string, causing it to display the invalid methodtype value where it should show valid options, and vice versa.

## Property-Based Test

```python
import pandas.errors
from hypothesis import given, strategies as st

class DummyClass:
    pass

@given(st.text(min_size=1).filter(lambda x: x not in {'method', 'classmethod', 'staticmethod', 'property'}))
def test_abstract_method_error_invalid_methodtype_message(invalid_methodtype):
    """
    Property: When AbstractMethodError is given an invalid methodtype,
    the error message should mention the invalid value and the valid options.
    Specifically, the error message should contain:
    1. The invalid methodtype value provided
    2. The set of valid options
    And the structure should be: "must be one of {valid_options}, got {invalid_value}"
    """
    valid_types = {'method', 'classmethod', 'staticmethod', 'property'}

    try:
        pandas.errors.AbstractMethodError(DummyClass(), methodtype=invalid_methodtype)
        assert False, f"Should have raised ValueError for invalid methodtype {invalid_methodtype}"
    except ValueError as e:
        error_msg = str(e)

        assert invalid_methodtype in error_msg, \
            f"Error message should mention the invalid value '{invalid_methodtype}', but got: {error_msg}"

        assert all(vtype in error_msg for vtype in valid_types), \
            f"Error message should mention all valid types {valid_types}, but got: {error_msg}"

        idx_invalid = error_msg.find(invalid_methodtype)
        idx_valid_set_start = error_msg.find('{')

        assert idx_valid_set_start < idx_invalid, \
            f"Error message structure is wrong: valid types should come before invalid value. Got: {error_msg}"

if __name__ == "__main__":
    test_abstract_method_error_invalid_methodtype_message()
```

<details>

<summary>
**Failing input**: `'0'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/17/hypo.py", line 20, in test_abstract_method_error_invalid_methodtype_message
    pandas.errors.AbstractMethodError(DummyClass(), methodtype=invalid_methodtype)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/errors/__init__.py", line 297, in __init__
    raise ValueError(
        f"methodtype must be one of {methodtype}, got {types} instead."
    )
ValueError: methodtype must be one of 0, got {'staticmethod', 'property', 'method', 'classmethod'} instead.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/17/hypo.py", line 38, in <module>
    test_abstract_method_error_invalid_methodtype_message()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/17/hypo.py", line 8, in test_abstract_method_error_invalid_methodtype_message
    def test_abstract_method_error_invalid_methodtype_message(invalid_methodtype):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/17/hypo.py", line 34, in test_abstract_method_error_invalid_methodtype_message
    assert idx_valid_set_start < idx_invalid, \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Error message structure is wrong: valid types should come before invalid value. Got: methodtype must be one of 0, got {'staticmethod', 'property', 'method', 'classmethod'} instead.
Falsifying example: test_abstract_method_error_invalid_methodtype_message(
    invalid_methodtype='0',  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import pandas.errors

class DummyClass:
    pass

try:
    pandas.errors.AbstractMethodError(DummyClass(), methodtype='foobar')
except ValueError as e:
    print(f"Error message: {e}")
```

<details>

<summary>
ValueError with incorrect error message format
</summary>
```
Error message: methodtype must be one of foobar, got {'method', 'property', 'staticmethod', 'classmethod'} instead.
```
</details>

## Why This Is A Bug

This bug violates the expected behavior of error messages in several ways:

1. **Standard Error Message Convention**: Error messages typically follow the pattern "must be one of {valid_options}, got {invalid_value} instead." This is a widely-used convention across programming languages and libraries.

2. **Variable Semantics**: The variable names in the source code clearly indicate the intended usage:
   - `types` is a set containing the valid methodtype values
   - `methodtype` is the parameter passed by the user

3. **User Experience Impact**: When a developer provides an invalid methodtype like 'foobar', they see "methodtype must be one of foobar" which incorrectly suggests that 'foobar' is a valid option. This creates confusion and wastes developer time.

4. **Code Intent**: The error message at line 298 in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/errors/__init__.py` clearly has the variables in the wrong position in the f-string.

While the pandas documentation doesn't explicitly specify the error message format, the intent is unambiguous from the variable names and standard programming practices.

## Relevant Context

The bug is located in the pandas errors module at `/home/npc/miniconda/lib/python3.13/site-packages/pandas/errors/__init__.py`, specifically in the `AbstractMethodError.__init__` method around line 298.

The `AbstractMethodError` class is used to signal that a method must be implemented in a subclass, and supports different method types: 'method', 'classmethod', 'staticmethod', and 'property'. When an invalid methodtype is provided, the class raises a ValueError with a message that should guide the developer to use one of the valid options.

Documentation reference: https://pandas.pydata.org/docs/reference/api/pandas.errors.AbstractMethodError.html

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