# Bug Report: pandas.core.window.rolling Malformed Error Message Due to Missing Quote

**Target**: `pandas.core.window.rolling.BaseWindow._validate`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The error message for invalid `method` parameter in pandas rolling window validation is missing a closing single quote, resulting in a malformed error message with unmatched quotes that appears unprofessional and potentially confusing to users.

## Property-Based Test

```python
import pandas as pd
from hypothesis import given, strategies as st, settings


@settings(max_examples=100)
@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=5, max_size=20),
    st.text(min_size=1, max_size=10).filter(lambda s: s not in ["single", "table"])
)
def test_method_validation_error_message(data, method):
    """
    Property: When an invalid method is provided, the error message should
    be well-formed with properly matched quotes.
    """
    df = pd.DataFrame({'A': data})

    try:
        df.rolling(window=2, method=method)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        error_msg = str(e)
        # Error message should have matched quotes
        assert error_msg.count("'") % 2 == 0, f"Unmatched quotes in error message: {error_msg}"


if __name__ == "__main__":
    test_method_validation_error_message()
```

<details>

<summary>
**Failing input**: `data=[0.0, 0.0, 0.0, 0.0, 0.0], method='0'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/62/hypo.py", line 18, in test_method_validation_error_message
    df.rolling(window=2, method=method)
    ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/generic.py", line 12599, in rolling
    return Rolling(
        self,
    ...<8 lines>...
        method=method,
    )
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/window/rolling.py", line 170, in __init__
    self._validate()
    ~~~~~~~~~~~~~~^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/window/rolling.py", line 1869, in _validate
    super()._validate()
    ~~~~~~~~~~~~~~~~~^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/window/rolling.py", line 208, in _validate
    raise ValueError("method must be 'table' or 'single")
ValueError: method must be 'table' or 'single

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/62/hypo.py", line 27, in <module>
    test_method_validation_error_message()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/62/hypo.py", line 6, in test_method_validation_error_message
    @given(

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/62/hypo.py", line 23, in test_method_validation_error_message
    assert error_msg.count("'") % 2 == 0, f"Unmatched quotes in error message: {error_msg}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Unmatched quotes in error message: method must be 'table' or 'single
Falsifying example: test_method_validation_error_message(
    data=[0.0, 0.0, 0.0, 0.0, 0.0],
    method='0',
)
```
</details>

## Reproducing the Bug

```python
import pandas as pd

df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})

try:
    df.rolling(window=2, method='invalid_method')
except ValueError as e:
    print(f"Error message: {e}")
    error_str = str(e)
    quote_count = error_str.count("'")
    print(f"Number of single quotes in error: {quote_count}")
```

<details>

<summary>
ValueError with malformed error message
</summary>
```
Error message: method must be 'table' or 'single
Number of single quotes in error: 3
```
</details>

## Why This Is A Bug

This violates basic string formatting expectations and professional standards for error messages in a major library like pandas. The error message contains exactly 3 single quotes (an odd number), clearly indicating unmatched quotes. The string literal starts with `"method must be 'table' or 'single` but is missing the closing single quote after 'single', making it appear as if the quote character is part of the valid value rather than a delimiter.

According to the pandas documentation for DataFrame.rolling(), the `method` parameter should accept only 'single' (default) or 'table' as valid values when using the numba engine. While the validation logic correctly enforces this constraint and raises a ValueError for invalid inputs, the error message itself is malformed due to a simple typo in the string literal at line 208 of `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/window/rolling.py`.

This is not a matter of interpretation or design choice - properly matched quotation marks are a fundamental requirement for well-formed strings in any programming context. The current state presents an objectively incorrect, unprofessional error message to users.

## Relevant Context

- **Location**: The bug is in the `_validate` method of `pandas.core.window.rolling.BaseWindow` class
- **File Path**: `/pandas/core/window/rolling.py`, line 208
- **pandas Documentation**: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html
- **Impact**: While the validation logic works correctly (correctly rejects invalid methods), the malformed error message looks unprofessional and could confuse users, especially those new to pandas
- **Affected versions**: Current version as of 2025-09-25

The `method` parameter is used to control execution mode when `engine='numba'`:
- 'single': executes the rolling operation per single column or row
- 'table': executes the rolling operation over the entire object

## Proposed Fix

```diff
--- a/pandas/core/window/rolling.py
+++ b/pandas/core/window/rolling.py
@@ -205,7 +205,7 @@ class BaseWindow(SelectionMixin):
                     f"the correct signature for get_window_bounds"
                 )
         if self.method not in ["table", "single"]:
-            raise ValueError("method must be 'table' or 'single")
+            raise ValueError("method must be 'table' or 'single'")
         if self.step is not None:
             if not is_integer(self.step):
                 raise ValueError("step must be an integer")
```