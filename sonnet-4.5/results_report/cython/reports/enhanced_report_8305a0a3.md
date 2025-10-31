# Bug Report: Cython.Utils.normalise_float_repr Produces Incorrect Results for Floats with Negative Exponents

**Target**: `Cython.Utils.normalise_float_repr`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `normalise_float_repr` function produces mathematically incorrect results when processing float strings with negative exponents and multiple significant digits, returning values that are orders of magnitude different from the input.

## Property-Based Test

```python
#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from Cython.Utils import normalise_float_repr

@given(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e100, max_value=1e100))
@settings(max_examples=1000)
def test_normalise_float_repr_round_trip(x):
    if x == 0.0:
        return

    float_str = str(x)
    normalized = normalise_float_repr(float_str)

    assert float(normalized) == float(float_str), (
        f"Round-trip failed: {float_str} -> {normalized} "
        f"({float(float_str)} != {float(normalized)})"
    )

if __name__ == "__main__":
    test_normalise_float_repr_round_trip()
```

<details>

<summary>
**Failing input**: `4.5930774792277246e-11`
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/19/hypo.py", line 23, in <module>
  |     test_normalise_float_repr_round_trip()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/19/hypo.py", line 9, in test_normalise_float_repr_round_trip
  |     @settings(max_examples=1000)
  |                    ^^^
  |   File "/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/19/hypo.py", line 17, in test_normalise_float_repr_round_trip
    |     assert float(normalized) == float(float_str), (
    |            ~~~~~^^^^^^^^^^^^
    | ValueError: could not convert string to float: '.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000-17091312394668117'
    | Falsifying example: test_normalise_float_repr_round_trip(
    |     x=-1.7091312394668117e-279,
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/19/hypo.py", line 17, in test_normalise_float_repr_round_trip
    |     assert float(normalized) == float(float_str), (
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    | AssertionError: Round-trip failed: 4.5930774792277246e-11 -> 4593077.00000000004792277246 (4.5930774792277246e-11 != 4593077.0)
    | Falsifying example: test_normalise_float_repr_round_trip(
    |     x=4.5930774792277246e-11,
    | )
    | Explanation:
    |     These lines were always and only run by failing examples:
    |         /home/npc/pbt/agentic-pbt/worker_/19/hypo.py:18
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Utils import normalise_float_repr

test_input = '1.192092896e-07'
result = normalise_float_repr(test_input)

print(f"Input:    {test_input}")
print(f"Output:   {result}")
print(f"Expected: .0000001192092896")
print()
print(f"Input float value:  {float(test_input)}")
print(f"Output float value: {float(result)}")
print()
print(f"Are they equal? {float(test_input) == float(result)}")
print()

# Let's also trace through the function manually for understanding
print("Manual trace:")
str_value = test_input.lower().lstrip('0')
print(f"  str_value after initial processing: '{str_value}'")

str_value_pre_exp, exp_str = str_value.split('e')
exp = int(exp_str)
print(f"  str_value before exp split: '{str_value_pre_exp}'")
print(f"  exp after parsing: {exp}")

# Process the decimal point
num_int_digits = str_value_pre_exp.index('.')
str_value_no_dot = str_value_pre_exp[:num_int_digits] + str_value_pre_exp[num_int_digits + 1:]
print(f"  num_int_digits: {num_int_digits}")
print(f"  str_value without dot: '{str_value_no_dot}'")

exp_adjusted = exp + num_int_digits
print(f"  exp after adjustment: {exp_adjusted}")

# The problematic calculation
print(f"  str_value_no_dot[:exp_adjusted] = str_value_no_dot[:{exp_adjusted}] = '{str_value_no_dot[:exp_adjusted]}'")
print(f"  str_value_no_dot[exp_adjusted:] = str_value_no_dot[{exp_adjusted}:] = '{str_value_no_dot[exp_adjusted:]}'")

# Assertion to show the failure
assert float(test_input) == float(result), f"Values don't match! {float(test_input)} != {float(result)}"
```

<details>

<summary>
AssertionError: Values don't match!
</summary>
```
Input:    1.192092896e-07
Output:   1192.000000092896
Expected: .0000001192092896

Input float value:  1.192092896e-07
Output float value: 1192.000000092896

Are they equal? False

Manual trace:
  str_value after initial processing: '1.192092896e-07'
  str_value before exp split: '1.192092896'
  exp after parsing: -7
  num_int_digits: 1
  str_value without dot: '1192092896'
  exp after adjustment: -6
  str_value_no_dot[:exp_adjusted] = str_value_no_dot[:-6] = '1192'
  str_value_no_dot[exp_adjusted:] = str_value_no_dot[-6:] = '092896'
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/19/repo.py", line 44, in <module>
    assert float(test_input) == float(result), f"Values don't match! {float(test_input)} != {float(result)}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Values don't match! 1.192092896e-07 != 1192.000000092896
```
</details>

## Why This Is A Bug

The function's docstring states it should "Generate a 'normalised', simple digits string representation of a float value to allow string comparisons." The test suite in `Tests/TestCythonUtils.py:198` explicitly requires that `float(result) == float(float_str)` must hold for all valid float strings.

The bug occurs when the exponent becomes negative after adjusting for the decimal point position. In the problematic line `str_value[:exp]` at line 680 of `/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Utils.py`, when `exp` is negative (e.g., -6), Python interprets this as a slice from the end of the string. For example, `'1192092896'[:-6]` returns `'1192'` (everything except the last 6 characters) instead of an empty string as the code logic intends.

This causes the function to produce completely incorrect results:
- For `1.192092896e-07`, it returns `1192.000000092896` (approximately 1192) instead of `.0000001192092896` (0.0000001192092896)
- The error magnitude is over 10 billion times (10^10)
- For very small numbers like `1.7091312394668117e-279`, the function produces invalid float strings containing negative signs in the middle that cannot be parsed

## Relevant Context

This function is used internally by the Cython compiler for float precision checking and normalization. The bug affects any float with:
1. A negative exponent in scientific notation
2. Multiple significant digits in the mantissa

The test suite already contains tests for this functionality at `/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Tests/TestCythonUtils.py` which verifies that normalized floats must maintain their numeric value.

This is not an edge case - scientific notation with negative exponents is the standard representation for small floating-point numbers in Python. The bug has been present in the slice logic that incorrectly handles negative indices.

## Proposed Fix

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
+            + '0' * max(0, exp - len(str_value))
+            + '.'
+            + str_value[exp:]
+        ).rstrip('0')

     return result if result != '.' else '.0'
```