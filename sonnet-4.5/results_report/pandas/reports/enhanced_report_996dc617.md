# Bug Report: pandas.core.dtypes.common.ensure_python_int Raises OverflowError Instead of TypeError for Infinity

**Target**: `pandas.core.dtypes.common.ensure_python_int`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `ensure_python_int` function violates its documented contract by raising `OverflowError` instead of `TypeError` when given infinity values, despite the docstring explicitly stating it should raise `TypeError` for any value that can't be converted to an integer.

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

if __name__ == "__main__":
    test_ensure_python_int_exception_contract()
```

<details>

<summary>
**Failing input**: `inf`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/47/hypo.py", line 7, in test_ensure_python_int_exception_contract
    result = ensure_python_int(value)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/common.py", line 116, in ensure_python_int
    new_value = int(value)
OverflowError: cannot convert float infinity to integer

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/47/hypo.py", line 18, in <module>
    test_ensure_python_int_exception_contract()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/47/hypo.py", line 5, in test_ensure_python_int_exception_contract
    def test_ensure_python_int_exception_contract(value):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/47/hypo.py", line 13, in test_ensure_python_int_exception_contract
    raise AssertionError(
        f"ensure_python_int raised OverflowError instead of TypeError for {value}"
    )
AssertionError: ensure_python_int raised OverflowError instead of TypeError for inf
Falsifying example: test_ensure_python_int_exception_contract(
    value=inf,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/47/hypo.py:12
```
</details>

## Reproducing the Bug

```python
from pandas.core.dtypes.common import ensure_python_int

# Test with positive infinity
print("Testing ensure_python_int with float('inf'):")
try:
    result = ensure_python_int(float('inf'))
    print(f"Result: {result}")
except OverflowError as e:
    print(f"Got OverflowError: {e}")
    print(f"Expected: TypeError according to docstring")
except TypeError as e:
    print(f"Got expected TypeError: {e}")

print()

# Test with negative infinity
print("Testing ensure_python_int with float('-inf'):")
try:
    result = ensure_python_int(float('-inf'))
    print(f"Result: {result}")
except OverflowError as e:
    print(f"Got OverflowError: {e}")
    print(f"Expected: TypeError according to docstring")
except TypeError as e:
    print(f"Got expected TypeError: {e}")
```

<details>

<summary>
OverflowError raised for both positive and negative infinity
</summary>
```
Testing ensure_python_int with float('inf'):
Got OverflowError: cannot convert float infinity to integer
Expected: TypeError according to docstring

Testing ensure_python_int with float('-inf'):
Got OverflowError: cannot convert float infinity to integer
Expected: TypeError according to docstring
```
</details>

## Why This Is A Bug

The function's docstring at lines 105-107 in `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/dtypes/common.py` explicitly states:

```
Raises
------
TypeError: if the value isn't an int or can't be converted to one.
```

The contract promises that `TypeError` will be raised for ANY value that can't be converted to an integer. However, when passed infinity values:

1. Line 109: The `is_float(value)` check passes since infinity is a valid float
2. Line 116: Python's built-in `int(float('inf'))` raises `OverflowError`
3. Line 118: The except clause only catches `(TypeError, ValueError, AssertionError)`
4. The `OverflowError` propagates uncaught, violating the documented contract

The function already demonstrates intent to normalize exceptions - it catches `ValueError` and `AssertionError` and re-raises them as `TypeError`. The failure to include `OverflowError` appears to be an oversight rather than intentional design.

## Relevant Context

This function is used throughout pandas for validating integer parameters, including in `pandas.core.indexes.range.RangeIndex.__new__` (lines 159-166), where it validates start, stop, and step parameters. Users writing error handling code based on the documented contract would get unexpected `OverflowError` exceptions instead of the promised `TypeError`.

The Python documentation states that `int()` raises `OverflowError` "if the argument is outside the range of a Python integer" which includes infinity values. This is a known behavior that the function should handle according to its contract.

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