# Bug Report: pandas.io.parsers validate_integer min_val Inconsistency

**Target**: `pandas.io.parsers.readers.validate_integer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `validate_integer` function inconsistently enforces the `min_val` constraint depending on whether the input is an integer or a float. Integer inputs are correctly validated against `min_val`, but float inputs that represent whole numbers bypass this check, allowing values below the minimum to pass validation.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pandas.io.parsers.readers import validate_integer

@settings(max_examples=1000)
@given(st.integers(max_value=-1), st.integers(min_value=0, max_value=100))
def test_validate_integer_min_val_consistency_int_vs_float(val, min_val):
    int_raised = False
    float_raised = False

    try:
        int_result = validate_integer("test", val, min_val=min_val)
    except ValueError:
        int_raised = True

    try:
        float_result = validate_integer("test", float(val), min_val=min_val)
    except ValueError:
        float_raised = True

    if val < min_val:
        assert int_raised, f"Integer {val} should raise ValueError"
        assert float_raised, f"Float {float(val)} should raise ValueError"
        assert int_raised == float_raised, "Inconsistent behavior"
```

**Failing input**: `val=-5, min_val=0`

## Reproducing the Bug

```python
from pandas.io.parsers.readers import validate_integer

try:
    result = validate_integer("test_param", -5, min_val=0)
    print(f"Integer -5: {result}")
except ValueError as e:
    print(f"Integer -5: ValueError - {e}")

try:
    result = validate_integer("test_param", -5.0, min_val=0)
    print(f"Float -5.0: {result}")
except ValueError as e:
    print(f"Float -5.0: ValueError - {e}")
```

**Output:**
```
Integer -5: ValueError - 'test_param' must be an integer >=0
Float -5.0: -5
```

## Why This Is A Bug

The function's docstring states it validates that the value is "an integer >={min_val}", and the `min_val` parameter documentation states "val < min_val will result in a ValueError". However, when a float representing a negative whole number is passed with a positive `min_val`, the function returns the converted integer without checking the minimum value constraint.

This violates the documented contract and creates inconsistent behavior between semantically equivalent inputs (`-5` vs `-5.0`).

## Fix

The bug is in the logic flow. When a float is detected, the code checks if it's a whole number and converts it, but then doesn't validate against `min_val`. The fix is to check `min_val` after converting the float to an integer:

```diff
--- a/pandas/io/parsers/readers.py
+++ b/pandas/io/parsers/readers.py
@@ -548,9 +548,11 @@ def validate_integer(
     msg = f"'{name:s}' must be an integer >={min_val:d}"
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