# Bug Report: pandas.io.parsers.readers validate_integer Skips Min Value Check for Floats

**Target**: `pandas.io.parsers.readers.validate_integer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `validate_integer` function incorrectly allows float values below the specified minimum value to pass validation when they can be losslessly converted to integers, violating its documented contract.

## Property-Based Test

```python
import pytest
from hypothesis import given, settings, strategies as st
from pandas.io.parsers.readers import validate_integer


@settings(max_examples=500)
@given(
    val=st.floats(allow_nan=False, allow_infinity=False),
    min_val=st.integers(min_value=0, max_value=1000)
)
def test_validate_integer_respects_min_val_for_floats(val, min_val):
    if val != int(val):
        return

    if int(val) >= min_val:
        result = validate_integer("test", val, min_val)
        assert result >= min_val
    else:
        with pytest.raises(ValueError, match="must be an integer"):
            validate_integer("test", val, min_val)

if __name__ == "__main__":
    # Run the test
    test_validate_integer_respects_min_val_for_floats()
```

<details>

<summary>
**Failing input**: `val=-1.0, min_val=0`
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/38
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_validate_integer_respects_min_val_for_floats FAILED        [100%]

=================================== FAILURES ===================================
______________ test_validate_integer_respects_min_val_for_floats _______________

    @settings(max_examples=500)
>   @given(

        val=st.floats(allow_nan=False, allow_infinity=False),
        min_val=st.integers(min_value=0, max_value=1000)
    )

hypo.py:7:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

val = -1.0, min_val = 0

    @settings(max_examples=500)
    @given(
        val=st.floats(allow_nan=False, allow_infinity=False),
        min_val=st.integers(min_value=0, max_value=1000)
    )
    def test_validate_integer_respects_min_val_for_floats(val, min_val):
        if val != int(val):
            return

        if int(val) >= min_val:
            result = validate_integer("test", val, min_val)
            assert result >= min_val
        else:
>           with pytest.raises(ValueError, match="must be an integer"):
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E           Failed: DID NOT RAISE <class 'ValueError'>
E           Falsifying example: test_validate_integer_respects_min_val_for_floats(
E               val=-1.0,
E               min_val=0,  # or any other generated value
E           )
E           Explanation:
E               These lines were always and only run by failing examples:
E                   /home/npc/pbt/agentic-pbt/worker_/38/hypo.py:19

hypo.py:19: Failed
=========================== short test summary info ============================
FAILED hypo.py::test_validate_integer_respects_min_val_for_floats - Failed: D...
============================== 1 failed in 0.62s ===============================
```
</details>

## Reproducing the Bug

```python
from pandas.io.parsers.readers import validate_integer

# Test case that should fail but doesn't
result = validate_integer("chunksize", -1.0, 1)
print(f"Result: {result}")
print(f"Type: {type(result)}")

# This should raise ValueError but returns -1 instead
print("\nTest passed when it should have raised ValueError!")
print(f"Expected: ValueError('chunksize' must be an integer >=1)")
print(f"Actual: Returned {result}")
```

<details>

<summary>
Returns -1 instead of raising ValueError
</summary>
```
Result: -1
Type: <class 'int'>

Test passed when it should have raised ValueError!
Expected: ValueError('chunksize' must be an integer >=1)
Actual: Returned -1
```
</details>

## Why This Is A Bug

The function's docstring explicitly states: "Minimum allowed value (val < min_val will result in a ValueError)". This contract is violated when `val` is a float that can be losslessly converted to an integer.

The bug occurs due to a logic error in the validation flow at lines 549-554 of `pandas/io/parsers/readers.py`. When a float value like `-1.0` is provided:
1. The code enters the `if is_float(val)` branch (line 549)
2. It checks if the float can be losslessly converted to int (line 550)
3. It converts the float to int (line 552)
4. **The code never checks if the converted value satisfies `val >= min_val`**
5. The `elif` branch containing the min value check is skipped entirely

In contrast, when an integer value like `-1` is provided, the code correctly enters the `elif` branch and validates both the type and the minimum value constraint, properly raising a ValueError.

This inconsistency means that `validate_integer("test", -1.0, 0)` incorrectly returns `-1`, while `validate_integer("test", -1, 0)` correctly raises a ValueError.

## Relevant Context

The `validate_integer` function is used internally by pandas to validate parameters in multiple places:
- `pandas/io/parsers/readers.py` line 1248: validates the `chunksize` parameter for `read_csv`
- `pandas/io/json/_json.py`: validates chunksize for JSON reading

This allows invalid parameters to be silently accepted by user-facing APIs. For example:
```python
import pandas as pd
import io

csv_data = "a,b,c\n1,2,3\n4,5,6"
reader = pd.read_csv(io.StringIO(csv_data), chunksize=-1.0)
# This succeeds when it should raise ValueError
# The reader produces empty chunks due to negative chunksize
```

The function documentation (lines 534-544) clearly indicates both requirements:
1. The function accepts "an integer OR float that can SAFELY be cast to an integer"
2. "val < min_val will result in a ValueError"

These are independent requirements that should both be enforced, but the current implementation only enforces the minimum value check for integer inputs, not for float inputs.

## Proposed Fix

```diff
--- a/pandas/io/parsers/readers.py
+++ b/pandas/io/parsers/readers.py
@@ -549,6 +549,8 @@ def validate_integer(
     if is_float(val):
         if int(val) != val:
             raise ValueError(msg)
         val = int(val)
+        if val < min_val:
+            raise ValueError(msg)
     elif not (is_integer(val) and val >= min_val):
         raise ValueError(msg)

     return int(val)
```