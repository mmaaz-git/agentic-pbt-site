# Bug Report: Cython.Utils.normalise_float_repr - Negative Numbers Produce Invalid Format

**Target**: `Cython.Utils.normalise_float_repr`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `normalise_float_repr` function incorrectly handles negative floating-point numbers, producing outputs with the minus sign in the wrong position (e.g., `.000-1` instead of `-.000001`), making them unparseable or having incorrect values.

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

**Failing input**: `f=-1e-05`

## Reproducing the Bug

```python
from Cython.Utils import normalise_float_repr

result = normalise_float_repr('-1e-05')
print(result)

try:
    value = float(result)
    print(f"Parsed value: {value}")
except ValueError as e:
    print(f"Cannot parse: {e}")
```

Output:
```
.000-1
Parsed value: -0.0001
```

The result `.000-1` is syntactically valid Python but semantically wrong (should be `.00001` or `-.00001`).

More severe example:
```python
result = normalise_float_repr('-2.6428581474819183e-115')
print(result)
float(result)
```

Output:
```
.00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000-26428581474819183
ValueError: could not convert string to float
```

## Why This Is A Bug

The function's purpose is to generate a normalized string representation of a float that preserves the numeric value. When handling negative numbers, the minus sign ends up in the fractional part instead of at the beginning, which:

1. Changes the value (`.000-1` parses as `0.000 - 1` = `-1` instead of `-0.0001`)
2. Produces unparseable strings for very small negative numbers
3. Violates the documented contract of producing "simple digits string representation"

The root cause is in the string manipulation logic around line 665-687 in Utils.py. When `str_value` contains a minus sign (e.g., `'-1'`) and we slice it with negative indices, the minus sign ends up in the wrong part of the output.

## Fix

```diff
--- a/Cython/Utils.py
+++ b/Cython/Utils.py
@@ -662,7 +662,12 @@ def normalise_float_repr(float_str):
     Generate a 'normalised', simple digits string representation of a float value
     to allow string comparisons.  Examples: '.123', '123.456', '123.'
     """
+    is_negative = float_str.lstrip().startswith('-')
     str_value = float_str.lower().lstrip('0')
+    if is_negative:
+        str_value = str_value.lstrip('-')
+        if not str_value or str_value == '.':
+            return '.0'

     exp = 0
     if 'E' in str_value or 'e' in str_value:
@@ -685,6 +690,9 @@ def normalise_float_repr(float_str):
     ).rstrip('0')

-    return result if result != '.' else '.0'
+    if result == '.':
+        result = '.0'
+
+    return ('-' + result) if is_negative else result
```