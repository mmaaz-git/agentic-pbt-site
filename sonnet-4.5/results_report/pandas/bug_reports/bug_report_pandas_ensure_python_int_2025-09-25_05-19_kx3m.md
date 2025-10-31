# Bug Report: pandas.core.dtypes.common.ensure_python_int Raises Wrong Exception Type

**Target**: `pandas.core.dtypes.common.ensure_python_int`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `ensure_python_int` function raises `OverflowError` instead of `TypeError` when given infinity values, violating its documented contract.

## Property-Based Test

```python
from pandas.core.dtypes.common import ensure_python_int
from hypothesis import given, strategies as st, settings
import pytest


@given(st.one_of(st.just(float('inf')), st.just(float('-inf')), st.just(float('nan'))))
@settings(max_examples=10)
def test_ensure_python_int_special_floats_raise_typeerror(x):
    with pytest.raises(TypeError):
        ensure_python_int(x)
```

**Failing input**: `inf`, `-inf`, `nan`

## Reproducing the Bug

```python
from pandas.core.dtypes.common import ensure_python_int

ensure_python_int(float('inf'))
```

```
OverflowError: cannot convert float infinity to integer
```

## Why This Is A Bug

The function's docstring explicitly states:

> Raises
> ------
> TypeError: if the value isn't an int or can't be converted to one.

However, when infinity or NaN values are provided, the function raises `OverflowError` (for inf) or `ValueError` (for NaN) instead of the documented `TypeError`. This is a contract violation - callers expecting to catch `TypeError` will not catch these special cases.

## Fix

```diff
--- a/pandas/core/dtypes/common.py
+++ b/pandas/core/dtypes/common.py
@@ -113,7 +113,7 @@ def ensure_python_int(value: int | np.integer) -> int:
             )
         raise TypeError(f"Wrong type {type(value)} for value {value}")
     try:
         new_value = int(value)
         assert new_value == value
-    except (TypeError, ValueError, AssertionError) as err:
+    except (TypeError, ValueError, AssertionError, OverflowError) as err:
         raise TypeError(f"Wrong type {type(value)} for value {value}") from err
     return new_value
```