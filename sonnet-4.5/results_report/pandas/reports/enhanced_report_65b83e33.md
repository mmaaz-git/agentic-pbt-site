# Bug Report: pandas.core.dtypes.common.ensure_python_int Raises Wrong Exception Type for Infinity Values

**Target**: `pandas.core.dtypes.common.ensure_python_int`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `ensure_python_int` function raises `OverflowError` instead of the documented `TypeError` when given positive or negative infinity values, violating its documented contract.

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


if __name__ == "__main__":
    # Run the test
    test_ensure_python_int_special_floats_raise_typeerror()
```

<details>

<summary>
**Failing input**: `inf`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/36/hypo.py", line 15, in <module>
    test_ensure_python_int_special_floats_raise_typeerror()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/36/hypo.py", line 7, in test_ensure_python_int_special_floats_raise_typeerror
    @settings(max_examples=10)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/36/hypo.py", line 10, in test_ensure_python_int_special_floats_raise_typeerror
    ensure_python_int(x)
    ~~~~~~~~~~~~~~~~~^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/common.py", line 116, in ensure_python_int
    new_value = int(value)
OverflowError: cannot convert float infinity to integer
Falsifying example: test_ensure_python_int_special_floats_raise_typeerror(
    x=inf,
)
```
</details>

## Reproducing the Bug

```python
from pandas.core.dtypes.common import ensure_python_int

# Test with positive infinity
try:
    result = ensure_python_int(float('inf'))
    print(f"Result for inf: {result}")
except Exception as e:
    print(f"Exception for inf: {type(e).__name__}: {e}")

# Test with negative infinity
try:
    result = ensure_python_int(float('-inf'))
    print(f"Result for -inf: {result}")
except Exception as e:
    print(f"Exception for -inf: {type(e).__name__}: {e}")

# Test with NaN
try:
    result = ensure_python_int(float('nan'))
    print(f"Result for nan: {result}")
except Exception as e:
    print(f"Exception for nan: {type(e).__name__}: {e}")
```

<details>

<summary>
OverflowError raised for infinity values instead of TypeError
</summary>
```
Exception for inf: OverflowError: cannot convert float infinity to integer
Exception for -inf: OverflowError: cannot convert float infinity to integer
Exception for nan: TypeError: Wrong type <class 'float'> for value nan
```
</details>

## Why This Is A Bug

The function's docstring explicitly states that it should raise `TypeError` "if the value isn't an int or can't be converted to one" (lines 106-107 of pandas/core/dtypes/common.py). Infinity values clearly "can't be converted to" an integer, so according to the documentation, they should raise `TypeError`.

The implementation already demonstrates intent to normalize exceptions - it catches `(TypeError, ValueError, AssertionError)` on line 118 and converts them all to `TypeError`. This works correctly for NaN values (which raise `ValueError` from `int(nan)`), but fails for infinity values because Python's `int()` raises `OverflowError` for infinity, which is not in the caught exception tuple.

This creates an inconsistency where:
- `float('nan')` → `TypeError` (correct, as documented)
- `float('inf')` → `OverflowError` (incorrect, should be `TypeError`)
- `float('-inf')` → `OverflowError` (incorrect, should be `TypeError`)

## Relevant Context

The `ensure_python_int` function is located in `pandas.core.dtypes.common`, which is part of pandas' private/internal API (not documented in the public API reference). While users shouldn't be importing this directly, the function is used internally within pandas, and having functions behave according to their docstrings is important for code maintainability.

The Python built-in `int()` function's behavior with special float values:
- `int(float('inf'))` → raises `OverflowError: cannot convert float infinity to integer`
- `int(float('-inf'))` → raises `OverflowError: cannot convert float infinity to integer`
- `int(float('nan'))` → raises `ValueError: cannot convert float NaN to integer`

The current implementation (lines 115-119) already handles `ValueError` by catching and re-raising as `TypeError`, showing clear intent for exception normalization. The omission of `OverflowError` appears to be an oversight.

## Proposed Fix

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