# Bug Report: Cython.Utils.normalise_float_repr Produces Invalid and Incorrect Output for Scientific Notation

**Target**: `Cython.Utils.normalise_float_repr`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `normalise_float_repr` function fails when processing numbers in scientific notation, producing either unparseable strings for negative values or incorrect numerical values for positive values with negative exponents.

## Property-Based Test

```python
import sys
from hypothesis import assume, given, strategies as st

sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Utils import normalise_float_repr


@given(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e50, max_value=1e50))
def test_normalise_float_repr_preserves_value(f):
    assume(f != 0.0 or str(f) != '-0.0')

    float_str = str(f)
    normalised = normalise_float_repr(float_str)

    original_value = float(float_str)
    normalised_value = float(normalised)

    assert original_value == normalised_value

if __name__ == "__main__":
    test_normalise_float_repr_preserves_value()
```

<details>

<summary>
**Failing input**: `f=-3.0608379956357947e-183` and `f=2.5176640922359985e-05`
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/61/hypo.py", line 22, in <module>
  |     test_normalise_float_repr_preserves_value()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/61/hypo.py", line 10, in test_normalise_float_repr_preserves_value
  |     def test_normalise_float_repr_preserves_value(f):
  |                    ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/61/hypo.py", line 17, in test_normalise_float_repr_preserves_value
    |     normalised_value = float(normalised)
    | ValueError: could not convert string to float: '.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000-30608379956357947'
    | Falsifying example: test_normalise_float_repr_preserves_value(
    |     f=-3.0608379956357947e-183,
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/61/hypo.py", line 19, in test_normalise_float_repr_preserves_value
    |     assert original_value == normalised_value
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    | AssertionError
    | Falsifying example: test_normalise_float_repr_preserves_value(
    |     f=2.5176640922359985e-05,
    | )
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Utils import normalise_float_repr

# Test case 1: Negative scientific notation that produces invalid output
test_value1 = "-1.938987928904224e-24"
print(f"Input: {test_value1}")
result1 = normalise_float_repr(test_value1)
print(f"Output: {result1!r}")

try:
    float_result1 = float(result1)
    print(f"Float conversion successful: {float_result1}")
except ValueError as e:
    print(f"Float conversion failed: {e}")

print()

# Test case 2: Positive scientific notation that produces incorrect value
test_value2 = "1.67660926681519e-08"
print(f"Input: {test_value2}")
result2 = normalise_float_repr(test_value2)
print(f"Output: {result2!r}")

try:
    float_result2 = float(result2)
    print(f"Float conversion successful: {float_result2}")
    original2 = float(test_value2)
    print(f"Original value: {original2}")
    print(f"Values match: {float_result2 == original2}")
except ValueError as e:
    print(f"Float conversion failed: {e}")

print()

# Test case 3: Simple negative number (for comparison - this works)
test_value3 = "-123.456"
print(f"Input: {test_value3}")
result3 = normalise_float_repr(test_value3)
print(f"Output: {result3!r}")

try:
    float_result3 = float(result3)
    print(f"Float conversion successful: {float_result3}")
    original3 = float(test_value3)
    print(f"Original value: {original3}")
    print(f"Values match: {float_result3 == original3}")
except ValueError as e:
    print(f"Float conversion failed: {e}")
```

<details>

<summary>
ValueError for negative scientific notation, wrong value for positive scientific notation
</summary>
```
Input: -1.938987928904224e-24
Output: '.0000000000000000000000-1938987928904224'
Float conversion failed: could not convert string to float: '.0000000000000000000000-1938987928904224'

Input: 1.67660926681519e-08
Output: '16766092.00000006681519'
Float conversion successful: 16766092.000000067
Original value: 1.67660926681519e-08
Values match: False

Input: -123.456
Output: '-123.456'
Float conversion successful: -123.456
Original value: -123.456
Values match: True
```
</details>

## Why This Is A Bug

The function fails to correctly handle the minus sign when processing scientific notation, leading to two critical issues:

1. **For negative numbers in scientific notation**: The minus sign gets misplaced during string manipulation, ending up in the middle of the number (e.g., `.0000000000000000000000-1938987928904224`) creating an unparseable string that raises `ValueError`.

2. **For positive numbers with negative exponents**: The function incorrectly interprets the negative exponent, treating it as if it were a positive exponent. For example, `1.67660926681519e-08` (which equals 0.0000000167660926681519) gets converted to `16766092.00000006681519` (approximately 16,766,092) - off by 15 orders of magnitude.

The function's docstring states it generates "a 'normalised', simple digits string representation of a float value to allow string comparisons" (Utils.py:661-663). The existing test suite (TestCythonUtils.py:195,198) explicitly verifies that `float(result) == float(original)`, which these failures violate.

## Relevant Context

The root cause is in the initial processing at Utils.py:665:
```python
str_value = float_str.lower().lstrip('0')
```

This line strips leading zeros but doesn't handle the minus sign separately. When the minus sign is present:
- For negative numbers: The minus remains, but later string slicing operations based on the exponent position cause the minus to end up in the wrong location
- The function doesn't distinguish between `-1.5e-08` (negative number, negative exponent) and `1.5e-08` (positive number, negative exponent)

The function is actively used in the Cython compiler (Compiler/ExprNodes.py) for processing float constants in DEF statements, making correct handling essential for compiler correctness.

Notably, simple negative numbers like `-123.456` work correctly because they don't involve exponent-based string manipulation where the sign positioning becomes corrupted.

## Proposed Fix

```diff
--- a/Utils.py
+++ b/Utils.py
@@ -662,7 +662,14 @@ def normalise_float_repr(float_str):
     Generate a 'normalised', simple digits string representation of a float value
     to allow string comparisons.  Examples: '.123', '123.456', '123.'
     """
-    str_value = float_str.lower().lstrip('0')
+    # Handle negative numbers by extracting and preserving the sign
+    sign = ''
+    str_value = float_str.lower()
+    if str_value.startswith('-'):
+        sign = '-'
+        str_value = str_value[1:]
+
+    str_value = str_value.lstrip('0')

     exp = 0
     if 'E' in str_value or 'e' in str_value:
@@ -677,7 +684,7 @@ def normalise_float_repr(float_str):
         num_int_digits = len(str_value)
     exp += num_int_digits

-    result = (
+    result = sign + (
         str_value[:exp]
         + '0' * (exp - len(str_value))
         + '.'
```