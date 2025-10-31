# Bug Report: Cython.Utils.normalise_float_repr - Exponent Calculation Error

**Target**: `Cython.Utils.normalise_float_repr`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `normalise_float_repr` function produces incorrect output values when handling floats with negative exponents and decimal points, resulting in values that differ by many orders of magnitude from the input.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from Cython.Utils import normalise_float_repr

@given(st.floats(allow_nan=False, allow_infinity=False))
def test_normalise_float_repr_value_preservation(f):
    float_str = str(f)
    result = normalise_float_repr(float_str)
    assert float(float_str) == float(result)
```

**Failing input**: `f=6.103515625e-05`

## Reproducing the Bug

```python
from Cython.Utils import normalise_float_repr

original = 6.103515625e-05
result = normalise_float_repr('6.103515625e-05')

print(f"Input:    6.103515625e-05")
print(f"Expected: .00006103515625 (approx)")
print(f"Got:      {result}")
print(f"")
print(f"Original value: {original}")
print(f"Result value:   {float(result)}")
print(f"Difference:     {abs(original - float(result))}")
```

Output:
```
Input:    6.103515625e-05
Expected: .00006103515625 (approx)
Got:      610351.00005625

Original value: 6.103515625e-05
Result value:   610351.00005625
Difference:     610351.00000521
```

The result is off by approximately 10 billion times the original value!

## Why This Is A Bug

The function's documented purpose is to produce a normalized string representation that preserves the float value. However, the exponent calculation logic on lines 667-677 is fundamentally flawed when handling numbers with both:
1. A decimal point in the mantissa
2. A negative exponent

The bug occurs because:
1. When parsing `'6.103515625e-05'`, the decimal point is removed: `str_value = '6103515625'`
2. `num_int_digits = 1` (the original position of the decimal point)
3. `exp = -5 + 1 = -4`
4. The result construction then uses `str_value[:exp]` which is `str_value[:-4]` = `'610351'`, placing the decimal point in completely the wrong position

The algorithm assumes `num_int_digits` represents where the decimal should go in the final number, but this assumption breaks down for negative exponents.

## Fix

The fix requires rethinking the exponent calculation logic to correctly handle negative exponents:

```diff
--- a/Cython/Utils.py
+++ b/Cython/Utils.py
@@ -672,13 +672,18 @@ def normalise_float_repr(float_str):

     if '.' in str_value:
         num_int_digits = str_value.index('.')
-        str_value = str_value[:num_int_digits] + str_value[num_int_digits + 1:]
+        str_value = str_value.replace('.', '')
     else:
         num_int_digits = len(str_value)
-    exp += num_int_digits
+
+    decimal_pos = exp + num_int_digits

     result = (
-        str_value[:exp]
-        + '0' * (exp - len(str_value))
+        str_value[:decimal_pos] if decimal_pos > 0 else ''
+    )
+    if decimal_pos < 0:
+        result += '0' * max(0, -decimal_pos - len(str_value))
+    result += (
         + '.'
-        + '0' * -exp
-        + str_value[exp:]
+        + '0' * max(0, -decimal_pos)
+        + str_value[max(0, decimal_pos):]
     ).rstrip('0')
```

Note: A complete fix would require more careful testing to ensure all edge cases are handled correctly.