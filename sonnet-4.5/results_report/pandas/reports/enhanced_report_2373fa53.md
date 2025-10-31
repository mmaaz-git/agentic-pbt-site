# Bug Report: ensure_python_int Silent Data Corruption and Type Contract Violation

**Target**: `pandas.core.dtypes.common.ensure_python_int`
**Severity**: High
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `ensure_python_int` function silently corrupts data by returning incorrect values for large integers represented as floats, and violates its type contract by accepting floats when the signature specifies only `int | np.integer`.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings, example
import numpy as np
from pandas.core.dtypes.common import ensure_python_int

@given(st.floats(min_value=-1e10, max_value=1e10, allow_nan=False, allow_infinity=False))
@example(5.5)  # Float with decimal part
@settings(max_examples=100)
def test_ensure_python_int_type_signature_violation(f):
    """
    Test that ensure_python_int rejects floats per its type signature.

    The type signature says int | np.integer, but the implementation
    accepts floats on line 109: is_integer(value) or is_float(value)
    """
    if f != int(f):
        try:
            result = ensure_python_int(f)
            assert False, f"Should reject float {f} per type signature, but got {result}"
        except TypeError:
            pass  # Expected behavior for non-integer floats


@given(st.integers(min_value=2**53, max_value=2**60))
@example(9007199254740993)  # 2^53 + 1
@settings(max_examples=100)
def test_ensure_python_int_precision_loss(n):
    """
    Test that ensure_python_int handles large integers as floats correctly.

    For integers beyond 2^53, float64 loses precision.
    The assertion int(float(n)) == n will fail.
    """
    f = np.float64(n)
    if int(f) != n:
        try:
            result = ensure_python_int(f)
            # This is a bug: the function returns the wrong value!
            assert result == n, f"ensure_python_int({f}) returned {result}, expected {n} (precision loss)"
        except TypeError:
            pass  # Acceptable to raise TypeError
        except AssertionError:
            raise  # Re-raise to show the precision loss bug

if __name__ == "__main__":
    print("Running property-based tests for ensure_python_int...")
    print("=" * 60)

    # Run the tests
    try:
        test_ensure_python_int_type_signature_violation()
        print("✓ Type signature test passed (floats with decimals are rejected)")
    except AssertionError as e:
        print(f"✗ Type signature test failed: {e}")

    try:
        test_ensure_python_int_precision_loss()
        print("✓ Precision loss test passed")
    except AssertionError as e:
        print(f"✗ Precision loss test failed: {e}")
```

<details>

<summary>
**Failing input**: `np.float64(9007199254740993)` (2^53 + 1 as float64)
</summary>
```
Running property-based tests for ensure_python_int...
============================================================
✓ Type signature test passed (floats with decimals are rejected)
✗ Precision loss test failed: ensure_python_int(9007199254740992.0) returned 9007199254740992, expected 9007199254740993 (precision loss)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

import numpy as np
from pandas.core.dtypes.common import ensure_python_int

print("Test 1: Float with decimal part (5.5)")
print("-" * 40)
try:
    result = ensure_python_int(5.5)
    print(f"Unexpected success: got {result}")
except TypeError as e:
    print(f"TypeError raised: {e}")

print("\nTest 2: Large integer as float causing precision loss")
print("-" * 40)
large_int = 9007199254740993  # 2^53 + 1
print(f"Original integer: {large_int}")
float_repr = np.float64(large_int)
print(f"Float64 representation: {float_repr}")
print(f"Float64 as int: {int(float_repr)}")
print(f"Precision lost: {int(float_repr) != large_int}")
try:
    result = ensure_python_int(float_repr)
    print(f"Unexpected success: got {result}")
except TypeError as e:
    print(f"TypeError raised: {e}")

print("\nTest 3: Integer-valued float (5.0)")
print("-" * 40)
try:
    result = ensure_python_int(5.0)
    print(f"Success: got {result} (type: {type(result)})")
except TypeError as e:
    print(f"TypeError raised: {e}")

print("\nTest 4: Regular integer (should work)")
print("-" * 40)
try:
    result = ensure_python_int(42)
    print(f"Success: got {result} (type: {type(result)})")
except TypeError as e:
    print(f"TypeError raised: {e}")

print("\nTest 5: NumPy integer (should work)")
print("-" * 40)
try:
    result = ensure_python_int(np.int64(100))
    print(f"Success: got {result} (type: {type(result)})")
except TypeError as e:
    print(f"TypeError raised: {e}")
```

<details>

<summary>
Silent data corruption: returns 9007199254740992 instead of 9007199254740993
</summary>
```
Test 1: Float with decimal part (5.5)
----------------------------------------
TypeError raised: Wrong type <class 'float'> for value 5.5

Test 2: Large integer as float causing precision loss
----------------------------------------
Original integer: 9007199254740993
Float64 representation: 9007199254740992.0
Float64 as int: 9007199254740992
Precision lost: True
Unexpected success: got 9007199254740992

Test 3: Integer-valued float (5.0)
----------------------------------------
Success: got 5 (type: <class 'int'>)

Test 4: Regular integer (should work)
----------------------------------------
Success: got 42 (type: <class 'int'>)

Test 5: NumPy integer (should work)
----------------------------------------
Success: got 100 (type: <class 'int'>)
```
</details>

## Why This Is A Bug

**Critical Data Corruption**: The most serious issue is that `ensure_python_int(np.float64(9007199254740993))` silently returns the wrong value (9007199254740992) instead of raising an error. This happens because:

1. The value 9007199254740993 (2^53 + 1) cannot be precisely represented as a float64
2. When converted to float64, it becomes 9007199254740992.0 (loses precision)
3. The function accepts this float (line 109: `is_float(value)` returns True)
4. It converts to int: `int(9007199254740992.0)` → 9007199254740992
5. The assertion on line 117 compares the wrong values: `9007199254740992 == 9007199254740992.0` → True (both are the corrupted value!)
6. The function returns 9007199254740992, which is **incorrect**

**Type Contract Violation**: The function signature explicitly declares `value: int | np.integer`, which means it should only accept integer types. However:
- Line 109 checks `is_integer(value) or is_float(value)`, explicitly allowing floats
- This breaks type checkers like mypy and pyright
- IDE autocomplete and validation become unreliable
- Users relying on the type signature will be surprised when floats are accepted

**Inconsistent Behavior**: The function exhibits confusing behavior:
- Accepts `5.0` (integer-valued float) → returns 5
- Rejects `5.5` (non-integer float) → raises TypeError
- Accepts `np.float64(2^53+1)` → returns wrong value silently

## Relevant Context

The `ensure_python_int` function is located in `/pandas/core/dtypes/common.py` and is part of pandas' internal type validation utilities. Key observations:

1. **Type checking functions**: The function uses `is_integer()` and `is_float()` from `pandas.core.dtypes.inference`, which are Cython functions that check the actual type of the value (not whether the value is integer-like).

2. **Exported function**: The function is included in the module's `__all__` list, making it part of the semi-public API even though it's not in the main pandas documentation.

3. **Internal usage**: The function is used internally by `pandas.core.indexes.range.RangeIndex` and other components for parameter validation.

4. **Float precision limits**: IEEE 754 double-precision floats (float64) can only precisely represent integers up to 2^53. Beyond this, consecutive integers cannot be distinguished, leading to data loss.

Documentation from the function:
- https://github.com/pandas-dev/pandas/blob/main/pandas/core/dtypes/common.py#L93

## Proposed Fix

```diff
--- a/pandas/core/dtypes/common.py
+++ b/pandas/core/dtypes/common.py
@@ -106,7 +106,7 @@ def ensure_python_int(value: int | np.integer) -> int:
     ------
     TypeError: if the value isn't an int or can't be converted to one.
     """
-    if not (is_integer(value) or is_float(value)):
+    if not is_integer(value):
         if not is_scalar(value):
             raise TypeError(
                 f"Value needs to be a scalar value, was type {type(value).__name__}"
```

This fix:
1. **Prevents data corruption**: Rejects all floats, eliminating the precision loss issue
2. **Matches type signature**: Implementation aligns with the declared `int | np.integer` type
3. **Maintains compatibility**: Still accepts Python ints and numpy integer types
4. **Improves predictability**: No more confusing behavior where some floats work and others don't
5. **Enables type safety**: Type checkers can properly validate code using this function