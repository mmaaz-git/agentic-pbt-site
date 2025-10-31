# Bug Report: pandas.core.dtypes.common.ensure_python_int OverflowError

**Target**: `pandas.core.dtypes.common.ensure_python_int`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `ensure_python_int` function violates its documented contract by raising `OverflowError` instead of `TypeError` when given infinity values. The docstring states it "Raises TypeError if the value isn't an int or can't be converted to one", but passing `float('inf')` or `float('-inf')` results in an uncaught `OverflowError`.

## Property-Based Test

```python
from pandas.core.dtypes.common import ensure_python_int
from hypothesis import given, strategies as st

@given(st.floats(allow_infinity=True))
def test_ensure_python_int_exception_contract(value):
    try:
        result = ensure_python_int(value)
        assert isinstance(result, int)
        assert result == value
    except TypeError:
        pass
    except OverflowError:
        raise AssertionError(
            f"ensure_python_int raised OverflowError instead of TypeError for {value}"
        )
```

**Failing input**: `float('inf')` or `float('-inf')`

## Reproducing the Bug

```python
from pandas.core.dtypes.common import ensure_python_int

try:
    ensure_python_int(float('inf'))
except OverflowError as e:
    print(f"Bug: Got OverflowError: {e}")
    print(f"Expected: TypeError according to docstring")

try:
    ensure_python_int(float('-inf'))
except OverflowError as e:
    print(f"Bug: Got OverflowError: {e}")
    print(f"Expected: TypeError according to docstring")
```

## Why This Is A Bug

The function's docstring (lines 93-108 in `pandas/core/dtypes/common.py`) explicitly states:

```
Raises
------
TypeError: if the value isn't an int or can't be converted to one.
```

However, when passed `float('inf')` or `float('-inf')`:
1. Line 109: The condition `is_float(value)` is True, so it passes the initial check
2. Line 116: `int(float('inf'))` is called, which raises `OverflowError`
3. Line 118: The except clause only catches `(TypeError, ValueError, AssertionError)`, not `OverflowError`
4. The `OverflowError` propagates up uncaught, violating the documented contract

This affects real usage in `pandas.core.indexes.range.RangeIndex.__new__` (line 159-166), where `ensure_python_int` is used to validate start, stop, and step parameters. Users would get an unexpected exception type when passing invalid infinity values.

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