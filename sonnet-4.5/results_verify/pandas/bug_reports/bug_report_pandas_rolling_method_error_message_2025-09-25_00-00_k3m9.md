# Bug Report: pandas.core.window.rolling Malformed Error Message

**Target**: `pandas.core.window.rolling.BaseWindow._validate`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The error message for invalid `method` parameter in rolling window validation has an unmatched quote, resulting in a malformed error message that is confusing to users.

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
```

**Failing input**: `method='invalid_method'` (or any string not in `["single", "table"]`)

## Reproducing the Bug

```python
import pandas as pd

df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})

try:
    df.rolling(window=2, method='invalid_method')
except ValueError as e:
    print(f"Error message: {e}")
```

Output:
```
Error message: method must be 'table' or 'single
```

## Why This Is A Bug

The error message `"method must be 'table' or 'single"` has unmatched quotes. The message starts with a double quote, contains `'table' or 'single`, and ends with a double quote. The second single quote (after `'single`) is missing, making the message confusing and appearing as if the quote is part of the valid value.

The correct message should be: `"method must be 'table' or 'single'"`

## Fix

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