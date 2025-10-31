# Bug Report: Cython.Utils normalise_float_repr Corrupts Negative Numbers

**Target**: `Cython.Utils.normalise_float_repr`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `normalise_float_repr` function fails to correctly handle negative numbers in scientific notation, producing either invalid float strings or dramatically corrupting the numeric value by treating the minus sign as part of the digit string.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import math
from Cython.Utils import normalise_float_repr

@given(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e100, max_value=1e100))
def test_normalise_float_repr_preserves_value(f):
    float_str = str(f)
    result = normalise_float_repr(float_str)
    assert math.isclose(float(result), float(float_str), rel_tol=1e-14)
```

**Failing input**: `f=-1.670758163823954e-133` and `f=1.114036198514633e-05`

## Reproducing the Bug

```python
from Cython.Utils import normalise_float_repr

print("Bug 1: Invalid float string for very small negative number")
f1 = -1.670758163823954e-133
float_str1 = str(f1)
result1 = normalise_float_repr(float_str1)
print(f"Input: {float_str1}")
print(f"Output: {result1}")
float(result1)

print("\nBug 2: Value corruption for small numbers")
f2 = 1.114036198514633e-05
float_str2 = str(f2)
result2 = normalise_float_repr(float_str2)
print(f"Input: {float_str2} = {f2}")
print(f"Output: {result2} = {float(result2)}")
print(f"Error: {abs(float(result2) - f2) / abs(f2) * 100:.1f}%")
```

## Why This Is A Bug

The function is documented to "generate a 'normalised', simple digits string representation of a float value to allow string comparisons" and should preserve the numeric value. However:

1. For `-1.670758163823954e-133`, it outputs `.00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000-1670758163823954` which is not a valid float
2. For `1.114036198514633e-05`, it outputs `111403619851.00004633`, changing the value by ~10^16 times

The root cause is that the function doesn't handle the minus sign correctly - it treats it as part of the digit string when calculating positions, causing the decimal point to be placed incorrectly.

## Fix

```diff
--- a/Cython/Utils.py
+++ b/Cython/Utils.py
@@ -662,7 +662,11 @@ def normalise_float_repr(float_str):
     Generate a 'normalised', simple digits string representation of a float value
     to allow string comparisons.  Examples: '.123', '123.456', '123.'
     """
-    str_value = float_str.lower().lstrip('0')
+    is_negative = float_str.startswith('-')
+    str_value = float_str.lstrip('-').lower().lstrip('0')
+    if not str_value or str_value[0] == '.':
+        str_value = '0' + str_value
+    if not str_value.replace('.', '').replace('e', ''):
+        str_value = '0'

     exp = 0
     if 'E' in str_value or 'e' in str_value:
@@ -684,4 +688,7 @@ def normalise_float_repr(float_str):
         + str_value[exp:]
     ).rstrip('0')

-    return result if result != '.' else '.0'
+    if result == '.':
+        result = '.0'
+
+    return ('-' if is_negative else '') + result
```