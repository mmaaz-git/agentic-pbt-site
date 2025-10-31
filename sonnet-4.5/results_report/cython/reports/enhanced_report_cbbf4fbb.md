# Bug Report: Cython.Utils.normalise_float_repr Mishandles Negative Numbers with Negative Exponents

**Target**: `Cython.Utils.normalise_float_repr`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `normalise_float_repr` function incorrectly processes negative numbers with negative exponents, producing invalid float strings with the minus sign embedded in the middle of the decimal representation instead of at the beginning.

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

if __name__ == "__main__":
    test_normalise_float_repr_value_preservation()
```

<details>

<summary>
**Failing input**: `-1.6845210745448747e-138`
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/3/hypo.py", line 13, in <module>
  |     test_normalise_float_repr_value_preservation()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/3/hypo.py", line 6, in test_normalise_float_repr_value_preservation
  |     @given(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e100, max_value=1e100))
  |                    ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/3/hypo.py", line 10, in test_normalise_float_repr_value_preservation
    |     assert math.isclose(float(normalized), float(float_str), rel_tol=1e-15)
    |                         ~~~~~^^^^^^^^^^^^
    | ValueError: could not convert string to float: '.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000-16845210745448747'
    | Falsifying example: test_normalise_float_repr_value_preservation(
    |     x=-1.6845210745448747e-138,
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/3/hypo.py", line 10, in test_normalise_float_repr_value_preservation
    |     assert math.isclose(float(normalized), float(float_str), rel_tol=1e-15)
    |            ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    | AssertionError
    | Falsifying example: test_normalise_float_repr_value_preservation(
    |     x=2.220446049250313e-16,
    | )
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
from Cython.Utils import normalise_float_repr

x = "-3.833509682449162e-128"
result = normalise_float_repr(x)
print(f"Input:  {x}")
print(f"Output: {result}")
print()

# Try to convert back to float
try:
    converted = float(result)
    print(f"Successfully converted back to float: {converted}")
except ValueError as e:
    print(f"ValueError: {e}")
```

<details>

<summary>
ValueError when converting malformed output back to float
</summary>
```
Input:  -3.833509682449162e-128
Output: .000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000-3833509682449162

ValueError: could not convert string to float: '.000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000-3833509682449162'
```
</details>

## Why This Is A Bug

The function violates its documented purpose of generating a "string representation of a float value" by producing malformed output that cannot be parsed as a valid float. The bug occurs at line 665 where `float_str.lower().lstrip('0')` strips leading zeros without preserving the minus sign for negative numbers.

For input like `-3.833509682449162e-128`, the function:
1. Converts to lowercase: `-3.833509682449162e-128`
2. Strips leading zeros from `-0.0000...3833509682449162`, which removes the zeros but orphans the minus sign
3. Produces `.0000...0000-3833509682449162` with the minus sign embedded after 128 zeros

This contradicts the function's contract because:
- The docstring states it generates a "string representation of a float value"
- The test suite validates that outputs can be converted back to float: `self.assertEqual(float(float_str), float(result))`
- A string with a minus sign in the middle is not a valid float representation in any programming language

## Relevant Context

The `normalise_float_repr` function is used internally within Cython's compiler infrastructure (specifically in ExprNodes.py) for handling float constants and allowing string comparisons of float values. While it appears to be an internal utility function without external documentation, it must handle the full range of valid float strings that Python can produce.

Key observations:
- The existing test suite in TestCythonUtils.py contains **zero negative number test cases**, which allowed this bug to remain undetected
- The bug affects **all negative numbers with negative exponents** when represented in scientific notation
- Negative numbers with positive exponents work correctly (e.g., `-5e10` → `-50000000000.`)
- The function is in `/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Utils.py`

## Proposed Fix

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
@@ -684,5 +689,8 @@ def normalise_float_repr(float_str):
         + str_value[exp:]
     ).rstrip('0')

-    return result if result != '.' else '.0'
+    result = result if result != '.' else '.0'
+    if is_negative and result != '.0':
+        result = '-' + result
+    return result
```