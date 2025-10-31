# Bug Report: pandas.core.dtypes.common.ensure_python_int Type Signature Violation and Precision Loss

**Target**: `pandas.core.dtypes.common.ensure_python_int`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `ensure_python_int` function violates its type signature by accepting float values when it explicitly declares `value: int | np.integer`, and silently returns incorrect values for large integers passed as float64 due to precision loss.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Hypothesis-based property tests for ensure_python_int bug.
"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
import numpy as np
from pandas.core.dtypes.common import ensure_python_int

@given(st.floats(min_value=-1e10, max_value=1e10, allow_nan=False, allow_infinity=False))
@settings(max_examples=100)
def test_ensure_python_int_type_signature_violation(f):
    """Test that floats violate the type signature."""
    if f != int(f):
        # Float with decimal part should be rejected
        try:
            result = ensure_python_int(f)
            assert False, f"Should reject float {f} per type signature, but got {result}"
        except TypeError:
            pass  # Expected
    else:
        # Integer-valued float still violates type signature
        try:
            result = ensure_python_int(f)
            # This succeeds but shouldn't according to type signature
            print(f"Type signature violation: float {f} accepted, returned {result}")
        except TypeError:
            pass


@given(st.integers(min_value=2**53, max_value=2**60))
@settings(max_examples=100)
def test_ensure_python_int_precision_loss(n):
    """Test precision loss for large integers converted to float."""
    f = np.float64(n)
    if int(f) != n:
        # Precision was lost in float conversion
        try:
            result = ensure_python_int(f)
            print(f"Precision loss bug: {n} -> float64 -> {int(f)} (lost {n - int(f)})")
            assert False, f"Should reject due to precision loss, but got {result}"
        except TypeError:
            pass  # Expected due to assertion failure
    else:
        # No precision loss
        try:
            result = ensure_python_int(f)
            assert result == n, f"Expected {n}, got {result}"
        except TypeError:
            # Still a type signature violation (float passed to int|np.integer function)
            pass

if __name__ == "__main__":
    print("Running hypothesis tests for ensure_python_int...")
    print("=" * 60)

    # Run the tests
    try:
        test_ensure_python_int_type_signature_violation()
        print("Type signature test completed")
    except AssertionError as e:
        print(f"Type signature test failed: {e}")

    try:
        test_ensure_python_int_precision_loss()
        print("Precision loss test completed")
    except AssertionError as e:
        print(f"Precision loss test failed: {e}")

    print("\nDemonstrating specific failing inputs:")
    print("-" * 40)

    # Specific failing case 1: Float with decimal
    print("\n1. Float with decimal part (5.5):")
    try:
        result = ensure_python_int(5.5)
        print(f"   ERROR: Got {result}, but should reject per type signature")
    except TypeError as e:
        print(f"   Raised TypeError: {e}")

    # Specific failing case 2: Large int with precision loss
    print("\n2. Large integer as float64 (9007199254740993):")
    large_int = 9007199254740993
    f = np.float64(large_int)
    print(f"   Original: {large_int}")
    print(f"   As float64: {f}")
    print(f"   Converted back: {int(f)}")
    try:
        result = ensure_python_int(f)
        print(f"   ERROR: Got {result}, lost precision!")
    except TypeError as e:
        print(f"   Raised TypeError: {e}")
```

<details>

<summary>
**Failing input**: `5.0` (type violation), `9007199254740993` (precision loss)
</summary>
```
Running hypothesis tests for ensure_python_int...
============================================================
Type signature violation: float 0.0 accepted, returned 0
Type signature violation: float -10000000000.0 accepted, returned -10000000000
Type signature violation: float 10000000000.0 accepted, returned 10000000000
Type signature violation: float 9999999999.0 accepted, returned 9999999999
Type signature test completed
Precision loss bug: 556920212827085093 -> float64 -> 556920212827085120 (lost -27)
Precision loss bug: 9007199254770249 -> float64 -> 9007199254770248 (lost 1)
Precision loss bug: 492690331495776102 -> float64 -> 492690331495776128 (lost -26)
Precision loss bug: 9007199254779099 -> float64 -> 9007199254779100 (lost -1)
Precision loss bug: 9007199254740993 -> float64 -> 9007199254740992 (lost 1)
Precision loss bug: 646153822266695147 -> float64 -> 646153822266695168 (lost -21)
Precision loss bug: 9007199254741089 -> float64 -> 9007199254741088 (lost 1)
Precision loss bug: 9007199254754601 -> float64 -> 9007199254754600 (lost 1)
Precision loss bug: 545031158678827639 -> float64 -> 545031158678827648 (lost -9)
Precision loss bug: 9007199254740993 -> float64 -> 9007199254740992 (lost 1)
Precision loss bug: 9007199254740993 -> float64 -> 9007199254740992 (lost 1)
Precision loss test failed: Should reject due to precision loss, but got 9007199254740992

Demonstrating specific failing inputs:
----------------------------------------

1. Float with decimal part (5.5):
   Raised TypeError: Wrong type <class 'float'> for value 5.5

2. Large integer as float64 (9007199254740993):
   Original: 9007199254740993
   As float64: 9007199254740992.0
   Converted back: 9007199254740992
   ERROR: Got 9007199254740992, lost precision!
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Minimal reproduction of ensure_python_int bug in pandas.
Demonstrates type signature violation and precision loss issues.
"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

import numpy as np
from pandas.core.dtypes.common import ensure_python_int

print("Testing ensure_python_int type signature and precision issues:")
print("=" * 60)

# Test 1: Float with decimal part (should reject per type signature but accepts)
print("\nTest 1: Float with decimal part")
print("Input: 5.5 (type: float)")
try:
    result = ensure_python_int(5.5)
    print(f"ERROR: Should have rejected float per type signature, but got: {result}")
except TypeError as e:
    print(f"TypeError raised (expected per assertion): {e}")

# Test 2: Integer-valued float (should reject per type signature but accepts)
print("\nTest 2: Integer-valued float")
print("Input: 5.0 (type: float)")
try:
    result = ensure_python_int(5.0)
    print(f"Result: {result} (type: {type(result).__name__})")
    print(f"Note: Type signature says 'int | np.integer' only, but float was accepted!")
except TypeError as e:
    print(f"TypeError: {e}")

# Test 3: Large integer losing precision when converted to float64
print("\nTest 3: Large integer precision loss")
large_int = 9007199254740993  # 2^53 + 1
print(f"Original integer: {large_int}")
float_version = np.float64(large_int)
print(f"As float64: {float_version}")
print(f"int(float64): {int(float_version)}")
print(f"Precision lost: {large_int != int(float_version)}")

try:
    result = ensure_python_int(float_version)
    print(f"ERROR: Should fail assertion, but got: {result}")
except TypeError as e:
    print(f"TypeError raised (expected): {e}")

# Test 4: Demonstrate the type signature violation more clearly
print("\nTest 4: Type signature analysis")
print(f"Function signature: ensure_python_int(value: int | np.integer) -> int")
print(f"Documentation says: 'value: int or numpy.integer'")
print(f"But implementation checks: 'is_integer(value) or is_float(value)'")
print(f"This violates the documented contract!")
```

<details>

<summary>
Output demonstrating the bug
</summary>
```
Testing ensure_python_int type signature and precision issues:
============================================================

Test 1: Float with decimal part
Input: 5.5 (type: float)
TypeError raised (expected per assertion): Wrong type <class 'float'> for value 5.5

Test 2: Integer-valued float
Input: 5.0 (type: float)
Result: 5 (type: int)
Note: Type signature says 'int | np.integer' only, but float was accepted!

Test 3: Large integer precision loss
Original integer: 9007199254740993
As float64: 9007199254740992.0
int(float64): 9007199254740992
Precision lost: True
ERROR: Should fail assertion, but got: 9007199254740992

Test 4: Type signature analysis
Function signature: ensure_python_int(value: int | np.integer) -> int
Documentation says: 'value: int or numpy.integer'
But implementation checks: 'is_integer(value) or is_float(value)'
This violates the documented contract!
```
</details>

## Why This Is A Bug

This is a contract violation bug with two critical issues:

1. **Type Signature Violation**: The function's type signature explicitly declares `value: int | np.integer`, and the docstring confirms "value: int or numpy.integer". However, line 109 of the implementation checks `is_integer(value) or is_float(value)`, accepting float values at runtime. This breaks the documented API contract and can cause type checking tools to miss errors.

2. **Silent Data Corruption**: For large integers beyond 2^53 (the limit of float64 precision), converting to float64 and back loses precision. The function returns incorrect values without warning. For example, `9007199254740993` becomes `9007199254740992` - a silent loss of data integrity.

3. **Inconsistent Behavior**: The function accepts some floats (5.0) but rejects others (5.5), creating confusing behavior not described in the documentation. The assertion `new_value == value` at line 117 catches floats with decimal parts but not precision loss cases.

## Relevant Context

The bug stems from conflicting design intentions. The `is_integer` and `is_float` helper functions from `pandas.core.dtypes.inference` check the actual Python type:
- `is_integer(5.0)` returns `False` (it's a float)
- `is_float(5.0)` returns `True`

The function appears to have been designed to accept integer-valued floats (like 5.0) but the type signature was never updated to reflect this. This creates a mismatch between the documented interface and runtime behavior.

Documentation: https://pandas.pydata.org/docs/reference/api/pandas.api.types.ensure_python_int.html
Source code: pandas/core/dtypes/common.py:93-120

## Proposed Fix

Remove float support to match the documented type signature:

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
- Makes the implementation match the type signature
- Prevents silent data corruption from precision loss
- Maintains backward compatibility with the documented API
- Ensures type checkers can properly validate code