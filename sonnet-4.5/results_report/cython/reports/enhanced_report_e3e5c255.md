# Bug Report: Cython.Utils normalise_float_repr Corrupts Negative Numbers

**Target**: `Cython.Utils.normalise_float_repr`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `normalise_float_repr` function fails to correctly handle negative numbers in scientific notation, producing either invalid float strings that cause ValueError exceptions or dramatically corrupting the numeric value by misplacing the decimal point.

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

<details>

<summary>
**Failing input**: `f=-5.6313238481951715e-223` and `f=2.220446049250313e-16`
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/8/hypo.py", line 12, in <module>
  |     test_normalise_float_repr_preserves_value()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/8/hypo.py", line 6, in test_normalise_float_repr_preserves_value
  |     def test_normalise_float_repr_preserves_value(f):
  |                    ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/8/hypo.py", line 9, in test_normalise_float_repr_preserves_value
    |     assert math.isclose(float(result), float(float_str), rel_tol=1e-14)
    |                         ~~~~~^^^^^^^^
    | ValueError: could not convert string to float: '.00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000-56313238481951715'
    | Falsifying example: test_normalise_float_repr_preserves_value(
    |     f=-5.6313238481951715e-223,
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/8/hypo.py", line 9, in test_normalise_float_repr_preserves_value
    |     assert math.isclose(float(result), float(float_str), rel_tol=1e-14)
    |            ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    | AssertionError
    | Falsifying example: test_normalise_float_repr_preserves_value(
    |     f=2.220446049250313e-16,
    | )
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
from Cython.Utils import normalise_float_repr

print("Bug 1: Invalid float string for very small negative number")
print("-" * 60)
f1 = -1.670758163823954e-133
float_str1 = str(f1)
result1 = normalise_float_repr(float_str1)
print(f"Input:  {float_str1}")
print(f"Output: {result1}")
print(f"Attempting to convert back to float...")
try:
    converted = float(result1)
    print(f"Successfully converted: {converted}")
except ValueError as e:
    print(f"ValueError: {e}")

print("\n" + "=" * 60 + "\n")

print("Bug 2: Value corruption for small numbers")
print("-" * 60)
f2 = 1.114036198514633e-05
float_str2 = str(f2)
result2 = normalise_float_repr(float_str2)
print(f"Input:  {float_str2} = {f2}")
print(f"Output: {result2} = {float(result2)}")
print(f"Error: {abs(float(result2) - f2) / abs(f2) * 100:.1f}%")

print("\n" + "=" * 60 + "\n")

print("Additional test cases")
print("-" * 60)
test_cases = [
    -1e-10,
    -1.0,
    -0.5,
    -10.5,
    -10000000000.0,
    0.0,
    1.0,
]

for f in test_cases:
    float_str = str(f)
    result = normalise_float_repr(float_str)
    print(f"Input: {float_str:20s} → Output: {result:20s}", end="")
    try:
        converted = float(result)
        if abs(converted - f) < 1e-10 or (f != 0 and abs((converted - f) / f) < 1e-10):
            print(" ✓ Correct")
        else:
            print(f" ✗ Wrong value: {converted}")
    except ValueError:
        print(" ✗ Invalid float")
```

<details>

<summary>
Output shows invalid float strings and massive value corruption
</summary>
```
Bug 1: Invalid float string for very small negative number
------------------------------------------------------------
Input:  -1.670758163823954e-133
Output: .00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000-1670758163823954
Attempting to convert back to float...
ValueError: could not convert string to float: '.00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000-1670758163823954'

============================================================

Bug 2: Value corruption for small numbers
------------------------------------------------------------
Input:  1.114036198514633e-05 = 1.114036198514633e-05
Output: 111403619851.00004633 = 111403619851.00005
Error: 999999999995841408.0%

============================================================

Additional test cases
------------------------------------------------------------
Input: -1e-10               → Output: .00000000-1          ✗ Invalid float
Input: -1.0                 → Output: -1.                  ✓ Correct
Input: -0.5                 → Output: -0.5                 ✓ Correct
Input: -10.5                → Output: -10.5                ✓ Correct
Input: -10000000000.0       → Output: -10000000000.        ✓ Correct
Input: 0.0                  → Output: .0                   ✓ Correct
Input: 1.0                  → Output: 1.                   ✓ Correct
```
</details>

## Why This Is A Bug

The function is documented to "generate a 'normalised', simple digits string representation of a float value to allow string comparisons" and is used in the Cython compiler for float value processing. The existing test suite in `/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Tests/TestCythonUtils.py` explicitly verifies that `float(result) == float(input)` (line 198), establishing the contract that the normalized output must preserve the numeric value.

However, the function violates this contract in two critical ways:

1. **Invalid Float Strings**: For negative numbers with very small exponents (e.g., `-1.670758163823954e-133`), it produces strings like `.00000000...000-1670758163823954` with the minus sign embedded in the middle of the digits. This is not valid Python float syntax and raises `ValueError` when converted back to float.

2. **Extreme Value Corruption**: For some numbers (e.g., `1.114036198514633e-05`), it produces valid float strings but with the decimal point misplaced, changing the value by factors of 10^16 (999999999995841408% error in the test case).

The root cause is in the function's logic at line 665: `str_value = float_str.lower().lstrip('0')`. When processing negative numbers, it strips the leading minus sign along with zeros, then treats the minus as part of the digit string when calculating decimal point positions. This causes the decimal point to be placed incorrectly in the final output construction (lines 679-685).

## Relevant Context

The function is located at `/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Utils.py:660-687`.

The test suite has no negative number test cases, which explains why this bug went undetected. All 21 test cases in `test_normalise_float_repr` (lines 169-202) use only positive numbers or zero.

The function is used in the Cython compiler's expression nodes (`Cython/Compiler/ExprNodes.py`) for compile-time float value processing and precision loss detection, making correctness critical.

Documentation: https://github.com/cython/cython

## Proposed Fix

```diff
--- a/Cython/Utils.py
+++ b/Cython/Utils.py
@@ -662,7 +662,12 @@ def normalise_float_repr(float_str):
     Generate a 'normalised', simple digits string representation of a float value
     to allow string comparisons.  Examples: '.123', '123.456', '123.'
     """
-    str_value = float_str.lower().lstrip('0')
+    # Handle negative numbers
+    is_negative = float_str.startswith('-')
+    if is_negative:
+        str_value = float_str[1:].lower().lstrip('0')
+    else:
+        str_value = float_str.lower().lstrip('0')

     exp = 0
     if 'E' in str_value or 'e' in str_value:
@@ -684,5 +689,8 @@ def normalise_float_repr(float_str):
         + str_value[exp:]
     ).rstrip('0')

-    return result if result != '.' else '.0'
+    if result == '.':
+        result = '.0'
+
+    return ('-' + result) if is_negative else result
```