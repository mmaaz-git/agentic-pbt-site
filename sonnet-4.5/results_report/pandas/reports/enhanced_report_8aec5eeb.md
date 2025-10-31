# Bug Report: pandas.errors.AbstractMethodError Swapped Error Message Arguments

**Target**: `pandas.errors.AbstractMethodError.__init__`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `AbstractMethodError.__init__` method has swapped arguments in its error message formatting, causing the invalid input to appear where valid options should be listed and vice versa, creating backwards and confusing error messages.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas.errors as pe


class DummyClass:
    pass


@given(st.text(min_size=1).filter(lambda x: x not in {"method", "classmethod", "staticmethod", "property"}))
def test_abstract_method_error_invalid_methodtype_message(invalid_methodtype):
    try:
        pe.AbstractMethodError(DummyClass(), methodtype=invalid_methodtype)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        error_msg = str(e)
        valid_types = {"method", "classmethod", "staticmethod", "property"}

        assert error_msg.index(str(valid_types)) < error_msg.index(invalid_methodtype), \
            f"Valid types should appear before invalid input in error message, got: {error_msg}"


if __name__ == "__main__":
    test_abstract_method_error_invalid_methodtype_message()
```

<details>

<summary>
**Failing input**: `'0'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/21/hypo.py", line 12, in test_abstract_method_error_invalid_methodtype_message
    pe.AbstractMethodError(DummyClass(), methodtype=invalid_methodtype)
    ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/errors/__init__.py", line 297, in __init__
    raise ValueError(
        f"methodtype must be one of {methodtype}, got {types} instead."
    )
ValueError: methodtype must be one of 0, got {'method', 'property', 'classmethod', 'staticmethod'} instead.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/21/hypo.py", line 23, in <module>
    test_abstract_method_error_invalid_methodtype_message()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/21/hypo.py", line 10, in test_abstract_method_error_invalid_methodtype_message
    def test_abstract_method_error_invalid_methodtype_message(invalid_methodtype):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/21/hypo.py", line 18, in test_abstract_method_error_invalid_methodtype_message
    assert error_msg.index(str(valid_types)) < error_msg.index(invalid_methodtype), \
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^
ValueError: substring not found
Falsifying example: test_abstract_method_error_invalid_methodtype_message(
    invalid_methodtype='0',  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import pandas.errors as pe


class DummyClass:
    pass


try:
    pe.AbstractMethodError(DummyClass(), methodtype="invalid_type")
except ValueError as e:
    print(f"Error message: {e}")
    print()
    print("Analysis:")
    print(f"  The error says 'methodtype must be one of invalid_type'")
    print(f"  But 'invalid_type' is the INVALID value we passed")
    print(f"  The valid values {{'method', 'classmethod', 'staticmethod', 'property'}} appear after 'got'")
    print()
    print("Expected format:")
    print("  methodtype must be one of {'method', 'classmethod', 'staticmethod', 'property'}, got invalid_type instead.")
```

<details>

<summary>
ValueError with backwards error message formatting
</summary>
```
Error message: methodtype must be one of invalid_type, got {'property', 'classmethod', 'method', 'staticmethod'} instead.

Analysis:
  The error says 'methodtype must be one of invalid_type'
  But 'invalid_type' is the INVALID value we passed
  The valid values {'method', 'classmethod', 'staticmethod', 'property'} appear after 'got'

Expected format:
  methodtype must be one of {'method', 'classmethod', 'staticmethod', 'property'}, got invalid_type instead.
```
</details>

## Why This Is A Bug

This violates Python's standard error message conventions where validation errors follow the pattern "parameter must be one of [valid options], got [invalid value] instead." The current implementation at line 298 of `/home/npc/miniconda/lib/python3.13/site-packages/pandas/errors/__init__.py` has the f-string template variables swapped, placing the invalid user input (`methodtype`) where the valid options should appear, and the valid options set (`types`) where the invalid input should be shown. This makes error messages confusing and harder to debug, as developers expect to see what values ARE valid first, followed by what they incorrectly provided. The backwards format contradicts patterns used throughout pandas and the broader Python ecosystem, where similar error messages correctly show valid options before invalid input.

## Relevant Context

The bug is in the `AbstractMethodError` class constructor which validates the `methodtype` parameter. This error class is used internally by pandas to indicate when abstract methods need to be implemented in concrete classes. While the error is correctly raised for invalid methodtypes, the message formatting makes it harder for developers to understand what went wrong.

The error message pattern used elsewhere in pandas follows the correct convention. For example:
- In `pandas/io/sql.py`: "engine must be one of 'auto', 'sqlalchemy'"
- In `pandas/io/parquet.py`: "engine must be one of 'pyarrow', 'fastparquet'"

Documentation: https://pandas.pydata.org/docs/reference/api/pandas.errors.AbstractMethodError.html
Source code: https://github.com/pandas-dev/pandas/blob/main/pandas/errors/__init__.py#L298

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