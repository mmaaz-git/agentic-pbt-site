# Bug Report: ensure_python_int raises OverflowError instead of TypeError

**Target**: `pandas.core.dtypes.common.ensure_python_int`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `ensure_python_int` function raises `OverflowError` when given infinity values instead of the documented `TypeError`. This violates the function's API contract which states it only raises `TypeError`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pandas.core.dtypes.common import ensure_python_int
import pytest


@given(st.floats(allow_nan=True, allow_infinity=True))
@settings(max_examples=200)
def test_ensure_python_int_special_floats(value):
    if np.isnan(value) or np.isinf(value):
        with pytest.raises(TypeError):
            ensure_python_int(value)
    elif value == int(value):
        result = ensure_python_int(value)
        assert result == int(value)
```

**Failing input**: `float('inf')`

## Reproducing the Bug

```python
from pandas.core.dtypes.common import ensure_python_int

ensure_python_int(float('inf'))
```

Output:
```
OverflowError: cannot convert float infinity to integer
```

Expected:
```
TypeError: Wrong type <class 'float'> for value inf
```

## Why This Is A Bug

The function's docstring (lines 93-107 in common.py) explicitly states:

```python
def ensure_python_int(value: int | np.integer) -> int:
    """
    Ensure that a value is a python int.

    ...

    Raises
    ------
    TypeError: if the value isn't an int or can't be converted to one.
    """
```

The function promises to only raise `TypeError`, but when given infinity, it raises `OverflowError` instead. This happens because the exception handler on line 118 only catches `TypeError`, `ValueError`, and `AssertionError`:

```python
try:
    new_value = int(value)
    assert new_value == value
except (TypeError, ValueError, AssertionError) as err:
    raise TypeError(f"Wrong type {type(value)} for value {value}") from err
```

However, `int(float('inf'))` raises `OverflowError`, which is not caught. This violates the API contract and could cause unexpected exceptions in calling code that only expects `TypeError`.

## Fix

```diff
diff --git a/pandas/core/dtypes/common.py b/pandas/core/dtypes/common.py
index 1234567..abcdefg 100644
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