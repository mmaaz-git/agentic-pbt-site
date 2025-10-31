# Bug Report: Cython.Utils.normalise_float_repr Negative Number Handling

**Target**: `Cython.Utils.normalise_float_repr`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `normalise_float_repr()` function produces invalid float string representations for negative numbers in scientific notation, causing the minus sign to appear in the wrong position.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from Cython.Utils import normalise_float_repr

@given(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e100, max_value=1e100))
@settings(max_examples=1000)
def test_normalise_float_repr_preserves_value(f):
    float_str = str(f)
    result = normalise_float_repr(float_str)

    original_value = float(float_str)
    result_value = float(result)

    assert original_value == result_value
```

**Failing input**: `-7.941487302529372e-299`

## Reproducing the Bug

```python
from Cython.Utils import normalise_float_repr

float_str = '-7.941487302529372e-299'
result = normalise_float_repr(float_str)

print(f"Input:  {float_str}")
print(f"Result: {result!r}")

result_value = float(result)
```

This raises `ValueError: could not convert string to float` because the result is:
`.000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000-7941487302529372`

The minus sign appears after the decimal point and zeros, making it an invalid float literal.

## Why This Is A Bug

The function's docstring states it should "Generate a 'normalised', simple digits string representation of a float value to allow string comparisons." The fundamental property that must hold is `float(input) == float(normalise_float_repr(input))`, which is violated for negative numbers in scientific notation. The existing unit tests in `TestCythonUtils.py` line 195 explicitly verify this property for positive numbers, but no negative test cases exist.

## Fix

```diff
--- a/Cython/Utils.py
+++ b/Cython/Utils.py
@@ -662,7 +662,13 @@ def normalise_float_repr(float_str):
     Generate a 'normalised', simple digits string representation of a float value
     to allow string comparisons.  Examples: '.123', '123.456', '123.'
     """
-    str_value = float_str.lower().lstrip('0')
+    # Extract sign
+    is_negative = False
+    str_value = float_str.lower().lstrip('0')
+    if str_value.startswith('-'):
+        is_negative = True
+        str_value = str_value[1:].lstrip('0')
+        if not str_value or str_value[0] == '.':
+            str_value = '0' + str_value

     exp = 0
     if 'E' in str_value or 'e' in str_value:
@@ -684,7 +690,10 @@ def normalise_float_repr(float_str):
         + str_value[exp:]
     ).rstrip('0')

-    return result if result != '.' else '.0'
+    if result == '.':
+        result = '.0'
+
+    return '-' + result if is_negative else result
```