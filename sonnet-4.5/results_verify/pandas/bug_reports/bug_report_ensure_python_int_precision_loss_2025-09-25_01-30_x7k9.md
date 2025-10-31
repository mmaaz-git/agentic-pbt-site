# Bug Report: ensure_python_int Silent Precision Loss

**Target**: `pandas.core.dtypes.common.ensure_python_int`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`ensure_python_int` silently accepts float values that have lost precision during conversion, returning an incorrect integer value instead of raising a TypeError as intended by the assertion at line 117.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
from pandas.core.dtypes.common import ensure_python_int

@given(st.integers(min_value=2**53 + 1, max_value=2**60))
def test_ensure_python_int_preserves_value(value):
    float_value = float(value)
    assume(float_value != value)

    result = ensure_python_int(float_value)
    assert result == value, f"ensure_python_int returned {result} instead of {value}"
```

**Failing input**: `value=9_007_199_254_740_993`, `float_value=9007199254740992.0`

## Reproducing the Bug

```python
from pandas.core.dtypes.common import ensure_python_int

original_value = 9_007_199_254_740_993
float_value = float(original_value)

print(f"Original integer:  {original_value}")
print(f"Float conversion:  {float_value}")
print(f"Precision lost:    {float_value != original_value}")

result = ensure_python_int(float_value)
print(f"\nResult: {result}")
print(f"Result matches original: {result == original_value}")
```

Output:
```
Original integer:  9007199254740993
Float conversion:  9007199254740992.0
Precision lost:    True

Result: 9007199254740992
Result matches original: False
```

## Why This Is A Bug

The function has an assertion `assert new_value == value` (line 117) that is intended to ensure the conversion is lossless. However, this assertion passes because it compares `int(float_value)` with `float_value`, both of which are already affected by the precision loss. The bug violates the documented behavior which states the function should raise TypeError "if the value isn't an int or can't be converted to one" - a float that loses precision cannot be correctly converted.

## Fix

The function should verify that the round-trip conversion preserves the original value before accepting it. One approach:

```diff
--- a/pandas/core/dtypes/common.py
+++ b/pandas/core/dtypes/common.py
@@ -114,7 +114,10 @@ def ensure_python_int(value: int | np.integer) -> int:
         raise TypeError(f"Wrong type {type(value)} for value {value}")
     try:
         new_value = int(value)
-        assert new_value == value
+        # Check that conversion is lossless by verifying round-trip
+        # For floats, this catches precision loss issues
+        if is_float(value):
+            assert float(new_value) == value and new_value == int(value)
     except (TypeError, ValueError, AssertionError) as err:
         raise TypeError(f"Wrong type {type(value)} for value {value}") from err
     return new_value
```

Alternatively, reject floats that exceed the safe integer range (2^53 for float64):

```diff
--- a/pandas/core/dtypes/common.py
+++ b/pandas/core/dtypes/common.py
@@ -108,6 +108,9 @@ def ensure_python_int(value: int | np.integer) -> int:
     """
     if not (is_integer(value) or is_float(value)):
         if not is_scalar(value):
             raise TypeError(
                 f"Value needs to be a scalar value, was type {type(value).__name__}"
             )
         raise TypeError(f"Wrong type {type(value)} for value {value}")
+    if is_float(value) and abs(value) > 2**53:
+        raise TypeError(f"Float value {value} exceeds safe integer range (2^53)")
     try:
         new_value = int(value)
         assert new_value == value
```