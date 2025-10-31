# Bug Report: ensure_python_int OverflowError Not Caught

**Target**: `pandas.core.dtypes.common.ensure_python_int`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `ensure_python_int` function does not catch `OverflowError` when converting infinity to an integer, violating its documented contract that it should raise `TypeError` for values that can't be converted to int.

## Property-Based Test

```python
from hypothesis import given, strategies as st, example
from pandas.core.dtypes.common import ensure_python_int
import pytest


@given(st.floats(allow_nan=True, allow_infinity=True))
@example(float('inf'))
@example(float('-inf'))
def test_ensure_python_int_raises_typeerror_for_invalid_floats(value):
    if value != int(value):
        with pytest.raises(TypeError):
            ensure_python_int(value)
```

**Failing input**: `float('inf')` and `float('-inf')`

## Reproducing the Bug

```python
from pandas.core.dtypes.common import ensure_python_int

try:
    ensure_python_int(float('inf'))
except TypeError as e:
    print(f"Correctly raised TypeError: {e}")
except OverflowError as e:
    print(f"BUG: Raised OverflowError instead of TypeError: {e}")
```

**Output:**
```
BUG: Raised OverflowError instead of TypeError: cannot convert float infinity to integer
```

## Why This Is A Bug

The function's docstring states:

```
Raises
------
TypeError: if the value isn't an int or can't be converted to one.
```

However, when called with `float('inf')` or `float('-inf')`, the function raises `OverflowError` instead of `TypeError`, violating its documented API contract.

Looking at the source code (pandas/core/dtypes/common.py:115-119):

```python
try:
    new_value = int(value)
    assert new_value == value
except (TypeError, ValueError, AssertionError) as err:
    raise TypeError(f"Wrong type {type(value)} for value {value}") from err
```

The function catches `TypeError`, `ValueError`, and `AssertionError`, but not `OverflowError`. When Python's `int()` is called on infinity, it raises `OverflowError`, which propagates uncaught.

## Fix

```diff
--- a/pandas/core/dtypes/common.py
+++ b/pandas/core/dtypes/common.py
@@ -115,7 +115,7 @@ def ensure_python_int(value: int | np.integer) -> int:
     try:
         new_value = int(value)
         assert new_value == value
-    except (TypeError, ValueError, AssertionError) as err:
+    except (TypeError, ValueError, AssertionError, OverflowError) as err:
         raise TypeError(f"Wrong type {type(value)} for value {value}") from err
     return new_value
```