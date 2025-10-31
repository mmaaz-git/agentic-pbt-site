# Bug Report: pandas.core.dtypes.common.ensure_python_int Raises OverflowError Instead of TypeError for Infinity Values

**Target**: `pandas.core.dtypes.common.ensure_python_int`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `ensure_python_int` function raises `OverflowError` instead of the documented `TypeError` when attempting to convert infinity values (float('inf') or float('-inf')) to integers, violating its API contract.

## Property-Based Test

```python
from hypothesis import given, strategies as st, example
from pandas.core.dtypes.common import ensure_python_int
import pytest


@given(st.floats(allow_nan=True, allow_infinity=True))
@example(float('inf'))
@example(float('-inf'))
def test_ensure_python_int_raises_typeerror_for_invalid_floats(value):
    try:
        int_value = int(value)
        if value != int_value:
            with pytest.raises(TypeError):
                ensure_python_int(value)
    except (OverflowError, ValueError):
        # For inf and nan, we expect TypeError from ensure_python_int
        with pytest.raises(TypeError):
            ensure_python_int(value)
```

<details>

<summary>
**Failing input**: `float('inf')` and `float('-inf')`
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/33
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_ensure_python_int_raises_typeerror_for_invalid_floats FAILED [100%]

=================================== FAILURES ===================================
__________ test_ensure_python_int_raises_typeerror_for_invalid_floats __________
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/33/hypo.py", line 7, in test_ensure_python_int_raises_typeerror_for_invalid_floats
  |     @example(float('inf'))
  |                    ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
  |     _raise_to_user(errors, state.settings, [], " in explicit examples")
  |     ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures in explicit examples. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/33/hypo.py", line 11, in test_ensure_python_int_raises_typeerror_for_invalid_floats
    |     int_value = int(value)
    | OverflowError: cannot convert float infinity to integer
    |
    | During handling of the above exception, another exception occurred:
    |
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/33/hypo.py", line 18, in test_ensure_python_int_raises_typeerror_for_invalid_floats
    |     ensure_python_int(value)
    |     ~~~~~~~~~~~~~~~~~^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/common.py", line 116, in ensure_python_int
    |     new_value = int(value)
    | OverflowError: cannot convert float infinity to integer
    | Falsifying explicit example: test_ensure_python_int_raises_typeerror_for_invalid_floats(
    |     value=inf,
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/33/hypo.py", line 11, in test_ensure_python_int_raises_typeerror_for_invalid_floats
    |     int_value = int(value)
    | OverflowError: cannot convert float infinity to integer
    |
    | During handling of the above exception, another exception occurred:
    |
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/33/hypo.py", line 18, in test_ensure_python_int_raises_typeerror_for_invalid_floats
    |     ensure_python_int(value)
    |     ~~~~~~~~~~~~~~~~~^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/common.py", line 116, in ensure_python_int
    |     new_value = int(value)
    | OverflowError: cannot convert float infinity to integer
    | Falsifying explicit example: test_ensure_python_int_raises_typeerror_for_invalid_floats(
    |     value=-inf,
    | )
    +------------------------------------
=========================== short test summary info ============================
FAILED hypo.py::test_ensure_python_int_raises_typeerror_for_invalid_floats - ...
============================== 1 failed in 0.34s ===============================
```
</details>

## Reproducing the Bug

```python
from pandas.core.dtypes.common import ensure_python_int

# Test with positive infinity
try:
    result = ensure_python_int(float('inf'))
    print(f"float('inf'): Converted successfully to {result}")
except TypeError as e:
    print(f"float('inf'): Correctly raised TypeError: {e}")
except OverflowError as e:
    print(f"float('inf'): BUG - Raised OverflowError instead of TypeError: {e}")

# Test with negative infinity
try:
    result = ensure_python_int(float('-inf'))
    print(f"float('-inf'): Converted successfully to {result}")
except TypeError as e:
    print(f"float('-inf'): Correctly raised TypeError: {e}")
except OverflowError as e:
    print(f"float('-inf'): BUG - Raised OverflowError instead of TypeError: {e}")

# Test with NaN for comparison
try:
    result = ensure_python_int(float('nan'))
    print(f"float('nan'): Converted successfully to {result}")
except TypeError as e:
    print(f"float('nan'): Correctly raised TypeError: {e}")
except OverflowError as e:
    print(f"float('nan'): BUG - Raised OverflowError instead of TypeError: {e}")
```

<details>

<summary>
Output demonstrating OverflowError instead of expected TypeError
</summary>
```
float('inf'): BUG - Raised OverflowError instead of TypeError: cannot convert float infinity to integer
float('-inf'): BUG - Raised OverflowError instead of TypeError: cannot convert float infinity to integer
float('nan'): Correctly raised TypeError: Wrong type <class 'float'> for value nan
```
</details>

## Why This Is A Bug

The function's docstring explicitly states that it will raise `TypeError` "if the value isn't an int or can't be converted to one." However, when called with infinity values, it raises `OverflowError` instead, violating this documented contract.

The implementation at pandas/core/dtypes/common.py:115-119 shows the function catches `TypeError`, `ValueError`, and `AssertionError` to re-raise them as `TypeError`, but fails to catch `OverflowError`:

```python
try:
    new_value = int(value)
    assert new_value == value
except (TypeError, ValueError, AssertionError) as err:
    raise TypeError(f"Wrong type {type(value)} for value {value}") from err
```

This implementation pattern demonstrates clear intent to provide a uniform `TypeError` interface for all conversion failures. The function already converts `ValueError` (raised for NaN) and `AssertionError` (raised for inexact floats like 5.5) into `TypeError`, showing the developers wanted consistent exception handling. The omission of `OverflowError` appears to be an oversight.

## Relevant Context

- The function is internal to pandas (`pandas.core.dtypes.common`) but is used throughout the pandas codebase
- Python's built-in `int()` function raises different exceptions for different failure modes:
  - `OverflowError` for infinity values
  - `ValueError` for NaN values
  - `TypeError` for incompatible types
- The function already handles `ValueError` correctly (converting NaN raises TypeError as expected)
- The function already handles inexact floats correctly (5.5 raises TypeError via AssertionError)
- Source code location: `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/dtypes/common.py:93-120`

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