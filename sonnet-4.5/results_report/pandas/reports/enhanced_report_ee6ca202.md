# Bug Report: pandas.errors.AbstractMethodError Swapped Variables in Error Message

**Target**: `pandas.errors.AbstractMethodError.__init__`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `AbstractMethodError.__init__` method contains swapped variables in its ValueError message formatting, causing the error to incorrectly display the invalid input where valid options should appear and vice versa.

## Property-Based Test

```python
import pandas.errors as pd_errors
from hypothesis import given, strategies as st, assume


@given(st.text(min_size=1))
def test_abstractmethoderror_error_message_property(invalid_methodtype):
    valid_types = {"method", "classmethod", "staticmethod", "property"}
    assume(invalid_methodtype not in valid_types)

    class DummyClass:
        pass

    try:
        pd_errors.AbstractMethodError(DummyClass(), methodtype=invalid_methodtype)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        error_msg = str(e)
        assert str(valid_types) in error_msg, f"Error should mention valid types, got: {error_msg}"
        assert f"got {invalid_methodtype}" in error_msg, f"Error should mention invalid input, got: {error_msg}"


if __name__ == "__main__":
    test_abstractmethoderror_error_message_property()
```

<details>

<summary>
**Failing input**: `invalid_methodtype='0'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/15/hypo.py", line 14, in test_abstractmethoderror_error_message_property
    pd_errors.AbstractMethodError(DummyClass(), methodtype=invalid_methodtype)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/errors/__init__.py", line 297, in __init__
    raise ValueError(
        f"methodtype must be one of {methodtype}, got {types} instead."
    )
ValueError: methodtype must be one of 0, got {'classmethod', 'property', 'staticmethod', 'method'} instead.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/15/hypo.py", line 23, in <module>
    test_abstractmethoderror_error_message_property()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/15/hypo.py", line 6, in test_abstractmethoderror_error_message_property
    def test_abstractmethoderror_error_message_property(invalid_methodtype):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/15/hypo.py", line 19, in test_abstractmethoderror_error_message_property
    assert f"got {invalid_methodtype}" in error_msg, f"Error should mention invalid input, got: {error_msg}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Error should mention invalid input, got: methodtype must be one of 0, got {'classmethod', 'property', 'staticmethod', 'method'} instead.
Falsifying example: test_abstractmethoderror_error_message_property(
    invalid_methodtype='0',  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import pandas.errors as pd_errors


class DummyClass:
    pass


# This should raise ValueError with swapped variables in error message
pd_errors.AbstractMethodError(DummyClass(), methodtype="foo")
```

<details>

<summary>
ValueError with incorrect variable placement
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/15/repo.py", line 9, in <module>
    pd_errors.AbstractMethodError(DummyClass(), methodtype="foo")
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/errors/__init__.py", line 297, in __init__
    raise ValueError(
        f"methodtype must be one of {methodtype}, got {types} instead."
    )
ValueError: methodtype must be one of foo, got {'property', 'method', 'staticmethod', 'classmethod'} instead.
```
</details>

## Why This Is A Bug

This violates the principle of clear error reporting by presenting information in a backwards manner. The error message format `"X must be one of {valid_options}, got {invalid_input} instead"` is a standard pattern in error messages across many libraries. The current implementation reverses this pattern by placing the user's invalid input ("foo") where the valid options should be displayed, and showing the set of valid options where the problematic input should appear.

The code clearly defines `types` as the set containing valid methodtype values (`{"method", "classmethod", "staticmethod", "property"}`), and `methodtype` is the parameter passed by the user. The validation check `if methodtype not in types:` confirms that `methodtype` should be validated against `types`, yet the error message has these variables reversed in the f-string on line 298.

This creates confusion for developers debugging their code, as they see an error claiming that "methodtype must be one of foo" (their invalid input) rather than seeing the actual valid options they should use. While the validation logic correctly prevents invalid values from being used, the misleading error message makes it harder for users to understand and fix their mistake.

## Relevant Context

The bug is located in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/errors/__init__.py` at lines 296-299. The AbstractMethodError class is designed to provide clearer error messages than the standard NotImplementedError for abstract methods that must be implemented in concrete classes.

The class accepts an optional `methodtype` parameter that must be one of four specific values: "method", "classmethod", "staticmethod", or "property". This parameter helps customize the error message based on the type of abstract method that wasn't implemented.

Documentation link: https://pandas.pydata.org/docs/reference/api/pandas.errors.AbstractMethodError.html
Source code: https://github.com/pandas-dev/pandas/blob/main/pandas/errors/__init__.py#L273-L309

The existing test suite (`pandas/tests/test_errors.py`) only tests valid methodtype values, which explains why this bug wasn't caught earlier. There are no tests for invalid methodtype values in the current test coverage.

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