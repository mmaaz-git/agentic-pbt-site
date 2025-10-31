# Bug Report: Cython.Utils.normalise_float_repr - Incorrect Exponent Handling

**Target**: `Cython.Utils.normalise_float_repr`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`normalise_float_repr` produces incorrect numeric values for numbers in scientific notation, dramatically changing the magnitude of the number.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from Cython.Utils import normalise_float_repr
import math

@settings(max_examples=1000)
@given(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e100, max_value=1e100))
def test_normalise_float_repr_value_preservation(x):
    float_str = str(x)
    normalized = normalise_float_repr(float_str)
    assert math.isclose(float(normalized), float(float_str), rel_tol=1e-15)
```

**Failing input**: `6.103515625e-05`

## Reproducing the Bug

```python
from Cython.Utils import normalise_float_repr

x = "6.103515625e-05"
result = normalise_float_repr(x)
print(f"Input:  {x}")
print(f"Output: {result}")
print(f"Input value:  {float(x)}")
print(f"Output value: {float(result)}")
```

**Output:**
```
Input:  6.103515625e-05
Output: 610351.00005625
Input value:  6.103515625e-05
Output value: 610351.00005625
```

The output is off by a factor of 10^10!

## Why This Is A Bug

The function is supposed to normalize float representations while preserving their numeric value (as evidenced by the test cases in TestCythonUtils.py line 195). However, the exponent handling logic on lines 667-677 is incorrect. The bug occurs because:

1. After splitting "6.103515625e-05" on 'e', we get `str_value = "6.103515625"` and `exp = -5`
2. The decimal point is removed, giving `str_value = "6103515625"` (length 10)
3. Line 677 adds `num_int_digits` (1) to `exp`, giving `exp = -4`
4. The algorithm then incorrectly interprets `exp = -4` as meaning "place decimal point 4 positions from the end"

This fundamentally misunderstands how scientific notation exponents work.

## Fix

The exponent handling needs to be redesigned. The current approach of tracking `num_int_digits` and modifying `exp` doesn't correctly handle negative exponents:

```diff
--- a/Cython/Utils.py
+++ b/Cython/Utils.py
@@ -670,9 +670,12 @@ def normalise_float_repr(float_str):
         exp = int(exp)

     if '.' in str_value:
         num_int_digits = str_value.index('.')
-        str_value = str_value[:num_int_digits] + str_value[num_int_digits + 1:]
+        num_frac_digits = len(str_value) - num_int_digits - 1
+        str_value = str_value.replace('.', '')
+        exp = exp - num_frac_digits
     else:
         num_int_digits = len(str_value)
-    exp += num_int_digits
+        exp = exp
+
+    exp += num_int_digits
```

Actually, the whole algorithm needs rethinking. A simpler approach would be to just convert through Python's float and back.