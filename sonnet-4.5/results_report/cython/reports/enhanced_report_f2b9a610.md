# Bug Report: Cython.Utils.normalise_float_repr Incorrect Handling of Scientific Notation with Negative Exponents

**Target**: `Cython.Utils.normalise_float_repr`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `normalise_float_repr` function produces incorrect results when normalizing floats in scientific notation with negative exponents. It generates values that are billions of times larger than the input for positive numbers, and produces unparseable strings with misplaced minus signs for negative numbers.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import math
from Cython.Utils import normalise_float_repr

@given(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e100, max_value=1e100))
def test_normalise_float_repr_round_trip(f):
    float_str = str(f)
    result = normalise_float_repr(float_str)
    assert math.isclose(float(result), f, rel_tol=1e-15)

if __name__ == "__main__":
    test_normalise_float_repr_round_trip()
```

<details>

<summary>
**Failing input**: `-2.6729003892890655e-68`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/54/hypo.py", line 12, in <module>
    test_normalise_float_repr_round_trip()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/54/hypo.py", line 6, in test_normalise_float_repr_round_trip
    def test_normalise_float_repr_round_trip(f):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/54/hypo.py", line 9, in test_normalise_float_repr_round_trip
    assert math.isclose(float(result), f, rel_tol=1e-15)
                        ~~~~~^^^^^^^^
ValueError: could not convert string to float: '.000000000000000000000000000000000000000000000000000000000000000000-26729003892890655'
Falsifying example: test_normalise_float_repr_round_trip(
    f=-2.6729003892890655e-68,
)
```
</details>

## Reproducing the Bug

```python
from Cython.Utils import normalise_float_repr

f1 = 5.960464477539063e-08
str1 = str(f1)
result1 = normalise_float_repr(str1)
print(f"Input: {f1}")
print(f"Input string: {str1}")
print(f"Result: {result1}")
try:
    parsed = float(result1)
    print(f"Float of result: {parsed}")
    if parsed != f1:
        print(f"BUG: Expected {f1}, got {parsed}")
except ValueError as e:
    print(f"BUG: Cannot parse result as float: {e}")

print()

f2 = -3.0929648190816446e-178
str2 = str(f2)
result2 = normalise_float_repr(str2)
print(f"Input: {f2}")
print(f"Input string: {str2}")
print(f"Result: {result2}")
try:
    parsed = float(result2)
    print(f"Float of result: {parsed}")
    if parsed != f2:
        print(f"BUG: Expected {f2}, got {parsed}")
except ValueError as e:
    print(f"BUG: Cannot parse result as float: {e}")
```

<details>

<summary>
Output shows incorrect normalization and unparseable results
</summary>
```
Input: 5.960464477539063e-08
Input string: 5.960464477539063e-08
Result: 596046447.00000007539063
Float of result: 596046447.0000001
BUG: Expected 5.960464477539063e-08, got 596046447.0000001

Input: -3.0929648190816446e-178
Input string: -3.0929648190816446e-178
Result: .00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000-30929648190816446
BUG: Cannot parse result as float: could not convert string to float: '.00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000-30929648190816446'
```
</details>

## Why This Is A Bug

The function violates its documented contract that "generate a 'normalised', simple digits string representation of a float value to allow string comparisons" (Cython/Utils.py:662-663). The existing test suite explicitly verifies that `float(float_str) == float(result)` for normalized outputs (Cython/Tests/TestCythonUtils.py:198), confirming that normalized strings must parse back to the same float value.

This bug breaks the fundamental property in two ways:

1. **For positive numbers with negative exponents** (e.g., 5.96e-08): The function produces a value ~10 billion times larger than the input. This happens because the function incorrectly calculates the position of the decimal point when the exponent is negative.

2. **For negative numbers with negative exponents** (e.g., -3.09e-178): The function produces an unparseable string with the minus sign embedded in the middle of zeros. This occurs because the function doesn't properly handle the sign separately from the numeric value.

The function is used in production code (Cython/Compiler/ExprNodes.py) for processing compile-time float constants, where precision and correctness are critical for the compiler's operation.

## Relevant Context

The bug is located in `/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Utils.py` at lines 660-688. The root cause is in the handling of the sign and exponent:

1. Line 665 calls `lstrip('0')` on the entire string including the minus sign, which causes issues with sign handling
2. Line 677 incorrectly calculates `exp += num_int_digits` without accounting for negative exponents properly
3. Lines 679-685 build the result string incorrectly when `exp` is negative, placing zeros in the wrong position

The existing test suite (TestCythonUtils.py:169-202) includes tests for scientific notation with both positive and negative exponents, showing this functionality is expected to work. The tests verify cases like '.1E-5' -> '.000001' and '123.456E-2' -> '1.23456', confirming that negative exponents should be supported.

## Proposed Fix

```diff
--- a/Cython/Utils.py
+++ b/Cython/Utils.py
@@ -662,7 +662,14 @@ def normalise_float_repr(float_str):
     Generate a 'normalised', simple digits string representation of a float value
     to allow string comparisons.  Examples: '.123', '123.456', '123.'
     """
-    str_value = float_str.lower().lstrip('0')
+    # Handle sign separately
+    is_negative = float_str.startswith('-')
+    if is_negative:
+        float_str = float_str[1:]
+
+    str_value = float_str.lower().lstrip('0')
+    if not str_value or str_value[0] == '.':
+        str_value = '0' + str_value if str_value else '0'

     exp = 0
     if 'E' in str_value or 'e' in str_value:
@@ -676,13 +683,25 @@ def normalise_float_repr(float_str):
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
+        # Small number with negative exponent
+        result = '.' + '0' * (-exp) + str_value
+    elif exp >= len(str_value):
+        # Large number, pad with zeros
+        result = str_value + '0' * (exp - len(str_value)) + '.'
+    else:
+        # Normal case, decimal in the middle
+        result = str_value[:exp] + '.' + str_value[exp:]
+
+    # Clean up trailing zeros after decimal point
+    if '.' in result:
+        result = result.rstrip('0')
+        if result.endswith('.'):
+            if result == '.':
+                result = '.0'
+
+    # Restore sign if negative
+    if is_negative:
+        result = '-' + result

     return result if result != '.' else '.0'
```