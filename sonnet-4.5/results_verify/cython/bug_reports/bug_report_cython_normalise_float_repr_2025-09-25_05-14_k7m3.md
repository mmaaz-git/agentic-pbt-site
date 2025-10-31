# Bug Report: Cython.Utils normalise_float_repr Scientific Notation

**Target**: `Cython.Utils.normalise_float_repr`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `normalise_float_repr` function in Cython.Utils produces completely incorrect normalized representations for floating-point numbers in scientific notation with negative exponents, resulting in values that are orders of magnitude different from the input.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from Cython.Utils import normalise_float_repr

@given(st.floats(allow_nan=False, allow_infinity=False))
def test_normalise_float_repr_round_trip(x):
    float_str = str(x)
    normalized = normalise_float_repr(float_str)
    assert float(normalized) == float(float_str)
```

**Failing input**: `x=1.192092896e-07`

## Reproducing the Bug

```python
from Cython.Utils import normalise_float_repr

test1 = "1.192092896e-07"
result1 = normalise_float_repr(test1)
print(f"Input:  {test1} = {float(test1)}")
print(f"Output: {result1} = {float(result1)}")

test2 = "6.25e-02"
result2 = normalise_float_repr(test2)
print(f"Input:  {test2} = {float(test2)}")
print(f"Output: {result2} = {float(result2)}")
```

**Output:**
```
Input:  1.192092896e-07 = 1.192092896e-07
Output: 1192.000000092896 = 1192.000000092896

Input:  6.25e-02 = 0.0625
Output: 62.05 = 62.05
```

## Why This Is A Bug

The function's purpose (from its docstring) is to "Generate a 'normalised', simple digits string representation of a float value to allow string comparisons." The examples given are `.123`, `123.456`, `123.`.

However, when given scientific notation with negative exponents, the function produces values that are completely different:
- `1.192092896e-07` (0.0000001192092896) becomes `1192.000000092896` (10 billion times larger!)
- `6.25e-02` (0.0625) becomes `62.05` (993 times larger!)

The bug occurs because the function uses negative Python slice indices when `exp` is negative, which doesn't correctly handle the case where the absolute value of the exponent is larger than the number of digits before the decimal point.

**Root cause** in lines 679-685 of `/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Utils.py`:

When processing `"1.192092896e-07"`:
1. After removing the decimal: `str_value = "1192092896"`, `exp = -6`
2. The code uses `str_value[:exp]` which becomes `str_value[:-6]` = `"1192"` (incorrect!)
3. Should place decimal point before all digits and add leading zeros, but instead constructs `"1192.000000092896"`

## Fix

```diff
--- a/Cython/Utils.py
+++ b/Cython/Utils.py
@@ -676,11 +676,16 @@ def normalise_float_repr(float_str):
         num_int_digits = len(str_value)
     exp += num_int_digits

-    result = (
-        str_value[:exp]
-        + '0' * (exp - len(str_value))
-        + '.'
-        + '0' * -exp
-        + str_value[exp:]
-    ).rstrip('0')
+    if exp <= 0:
+        result = (
+            '.'
+            + '0' * -exp
+            + str_value
+        ).rstrip('0')
+    else:
+        result = (
+            str_value[:exp]
+            + '0' * (exp - len(str_value))
+            + '.'
+            + str_value[exp:]
+        ).rstrip('0')

     return result if result != '.' else '.0'
```