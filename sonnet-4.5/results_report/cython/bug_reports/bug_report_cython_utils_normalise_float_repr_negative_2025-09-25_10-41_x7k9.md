# Bug Report: Cython.Utils.normalise_float_repr - Incorrect Handling of Negative Numbers

**Target**: `Cython.Utils.normalise_float_repr`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `normalise_float_repr` function produces syntactically invalid float strings when given negative numbers with negative exponents. The minus sign is placed in the wrong position within the normalized output, making the result unparseable as a float.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from Cython.Utils import normalise_float_repr


@given(st.floats(allow_nan=False, allow_infinity=False,
                 min_value=-1e50, max_value=1e50))
@settings(max_examples=1000)
def test_round_trip_property(f):
    """
    Property: normalise_float_repr should preserve the numerical value.
    For any valid float string, the normalized form should represent the same number.
    """
    float_str = str(f)
    result = normalise_float_repr(float_str)

    assert float(result) == float(float_str), \
        f"Value changed: {float_str} -> {result} ({float(float_str)} != {float(result)})"
```

**Failing input**: `-1.1754943508222875e-38`

## Reproducing the Bug

```python
from Cython.Utils import normalise_float_repr

test_cases = [
    "-1e-10",
    "-0.00001",
    "-1.5e-5",
    "-1.1754943508222875e-38",
]

for float_str in test_cases:
    result = normalise_float_repr(float_str)
    print(f"{float_str:30} -> {result}")
    try:
        float(result)
    except ValueError:
        print(f"  ERROR: '{result}' is not a valid float!")
```

**Output**:
```
-1e-10                         -> .00000000-1
  ERROR: '.00000000-1' is not a valid float!
-0.00001                       -> .0000-1
  ERROR: '.0000-1' is not a valid float!
-1.5e-5                        -> .0000-15
  ERROR: '.0000-15' is not a valid float!
-1.1754943508222875e-38        -> .000000000000000000000000000000000000-11754943508222875
  ERROR: '.000000000000000000000000000000000000-11754943508222875' is not a valid float!
```

## Why This Is A Bug

The function is documented to "Generate a 'normalised', simple digits string representation of a float value to allow string comparisons." However, it produces syntactically invalid output for negative numbers.

The root cause is that the function does not handle the minus sign separately from the digit manipulation logic. When processing a negative number like `-1e-10`, the minus sign becomes embedded within the digit string and ends up in the wrong position after the decimal point manipulation.

Expected output for `-1e-10` should be `-.0000000001` or similar, but actual output is `.00000000-1`.

## Fix

The function needs to extract and preserve the sign separately before processing the digits:

```diff
 def normalise_float_repr(float_str):
     """
     Generate a 'normalised', simple digits string representation of a float value
     to allow string comparisons.  Examples: '.123', '123.456', '123.'
     """
-    str_value = float_str.lower().lstrip('0')
+    str_value = float_str.lower()
+
+    # Handle negative sign separately
+    sign = ''
+    if str_value.startswith('-'):
+        sign = '-'
+        str_value = str_value[1:]
+
+    # Strip leading zeros after extracting sign
+    str_value = str_value.lstrip('0')

     exp = 0
     if 'E' in str_value or 'e' in str_value:
         str_value, exp = str_value.split('E' if 'E' in str_value else 'e', 1)
         exp = int(exp)

     if '.' in str_value:
         num_int_digits = str_value.index('.')
         str_value = str_value[:num_int_digits] + str_value[num_int_digits + 1:]
     else:
         num_int_digits = len(str_value)
     exp += num_int_digits

     result = (
         str_value[:exp]
         + '0' * (exp - len(str_value))
         + '.'
         + '0' * -exp
         + str_value[exp:]
     ).rstrip('0')

-    return result if result != '.' else '.0'
+    normalized = result if result != '.' else '.0'
+    return sign + normalized
```