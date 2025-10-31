# Bug Report: Cython.Utils.normalise_float_repr Misplaces Minus Sign for Negative Scientific Notation

**Target**: `Cython.Utils.normalise_float_repr`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `normalise_float_repr()` function incorrectly handles negative numbers in scientific notation with negative exponents, placing the minus sign in the middle of the normalized string rather than at the beginning, resulting in unparseable output.

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

# Run the test
if __name__ == "__main__":
    test_normalise_float_repr_preserves_value()
```

<details>

<summary>
**Failing input**: `-2.9993712029908333e-298`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/49/hypo.py", line 17, in <module>
    test_normalise_float_repr_preserves_value()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/49/hypo.py", line 5, in test_normalise_float_repr_preserves_value
    @settings(max_examples=1000)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/49/hypo.py", line 11, in test_normalise_float_repr_preserves_value
    result_value = float(result)
ValueError: could not convert string to float: '.00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000-29993712029908333'
Falsifying example: test_normalise_float_repr_preserves_value(
    f=-2.9993712029908333e-298,
)
```
</details>

## Reproducing the Bug

```python
from Cython.Utils import normalise_float_repr

# Test case from the bug report
float_str = '-7.941487302529372e-299'
result = normalise_float_repr(float_str)

print(f"Input:  {float_str}")
print(f"Result: {result!r}")

# Try to parse it back
try:
    result_value = float(result)
    print(f"Parsed value: {result_value}")
except ValueError as e:
    print(f"Error parsing result: {e}")
```

<details>

<summary>
ValueError: Minus sign appears after decimal point and zeros instead of at the beginning
</summary>
```
Input:  -7.941487302529372e-299
Result: '.000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000-7941487302529372'
Error parsing result: could not convert string to float: '.000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000-7941487302529372'
```
</details>

## Why This Is A Bug

This violates the fundamental contract established by the function's test suite in `TestCythonUtils.py` line 198, which explicitly verifies that `float(float_str) == float(normalise_float_repr(float_str))`. The function's docstring states it should "Generate a 'normalised', simple digits string representation of a float value to allow string comparisons," but the output cannot even be parsed back to a float.

The bug occurs because the function treats the minus sign as a digit when calculating string positions. When processing `-7.941487302529372e-299`:
1. The input is initially parsed as `str_value = '-7.941487302529372'` with `exp = -299`
2. The function incorrectly counts `num_int_digits = 2` (including the minus sign as a digit position)
3. This causes the exponent adjustment to be wrong: `exp = -299 + 2 = -297`
4. During string construction, the minus sign ends up embedded within the zeros rather than at the beginning

The test suite only includes positive number test cases, missing this critical edge case for negative numbers with negative exponents.

## Relevant Context

- **Function location**: `/Cython/Utils.py` lines 660-687
- **Test location**: `/Cython/Tests/TestCythonUtils.py` lines 169-202
- **Usage context**: Used in `Cython/Compiler/ExprNodes.py` for comparing float representations and detecting precision loss
- **Affected inputs**: Only negative numbers in scientific notation with negative exponents (e.g., `-1e-10`, `-7.94e-299`)
- **Working cases**: Positive numbers, negative numbers without exponents, negative numbers with positive exponents

The function works correctly for:
- Positive numbers: `1.23e-10` → `.000000000123`
- Negative integers: `-123` → `-123.`
- Negative decimals: `-0.5` → `-.5`
- Negative with positive exponent: `-1e10` → `-10000000000.`

## Proposed Fix

```diff
--- a/Cython/Utils.py
+++ b/Cython/Utils.py
@@ -662,7 +662,13 @@ def normalise_float_repr(float_str):
     Generate a 'normalised', simple digits string representation of a float value
     to allow string comparisons.  Examples: '.123', '123.456', '123.'
     """
-    str_value = float_str.lower().lstrip('0')
+    # Handle negative numbers
+    is_negative = float_str.startswith('-')
+    if is_negative:
+        float_str = float_str[1:]
+
+    # Remove leading zeros
+    str_value = float_str.lower().lstrip('0')

     exp = 0
     if 'E' in str_value or 'e' in str_value:
@@ -684,4 +690,8 @@ def normalise_float_repr(float_str):
         + str_value[exp:]
     ).rstrip('0')

-    return result if result != '.' else '.0'
+    if result == '.':
+        result = '.0'
+
+    # Add back the negative sign at the beginning
+    return ('-' + result) if is_negative else result
```