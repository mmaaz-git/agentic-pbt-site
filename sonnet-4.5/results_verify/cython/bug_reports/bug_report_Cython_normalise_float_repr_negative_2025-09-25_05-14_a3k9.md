# Bug Report: Cython.Utils.normalise_float_repr - Negative Numbers with Exponents

**Target**: `Cython.Utils.normalise_float_repr`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`normalise_float_repr` produces invalid float strings for negative numbers in scientific notation, placing the minus sign in the wrong position.

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

**Failing input**: `-3.833509682449162e-128`

## Reproducing the Bug

```python
from Cython.Utils import normalise_float_repr

x = "-3.833509682449162e-128"
result = normalise_float_repr(x)
print(f"Input:  {x}")
print(f"Output: {result}")

float(result)
```

**Output:**
```
Input:  -3.833509682449162e-128
Output: .000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000-3833509682449162
ValueError: could not convert string to float: '.000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000-3833509682449162'
```

## Why This Is A Bug

The minus sign ends up embedded in the string after the zeros rather than at the beginning. This violates the function's contract of producing a valid float string representation. The issue is on line 665 where `.lstrip('0')` is called on the lowercased string - for negative numbers like `-0.00...`, this removes leading zeros without properly handling the minus sign.

## Fix

```diff
--- a/Cython/Utils.py
+++ b/Cython/Utils.py
@@ -662,7 +662,12 @@ def normalise_float_repr(float_str):
     Generate a 'normalised', simple digits string representation of a float value
     to allow string comparisons.  Examples: '.123', '123.456', '123.'
     """
-    str_value = float_str.lower().lstrip('0')
+    str_value = float_str.lower()
+    is_negative = str_value.startswith('-')
+    if is_negative:
+        str_value = str_value[1:]
+
+    str_value = str_value.lstrip('0')

     exp = 0
     if 'E' in str_value or 'e' in str_value:
@@ -684,7 +689,10 @@ def normalise_float_repr(float_str):
         + str_value[exp:]
     ).rstrip('0')

-    return result if result != '.' else '.0'
+    result = result if result != '.' else '.0'
+    if is_negative and result != '.0':
+        result = '-' + result
+    return result
```