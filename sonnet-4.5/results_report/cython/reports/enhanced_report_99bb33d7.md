# Bug Report: Cython.Utils.normalise_float_repr Multiple Critical Errors with Scientific Notation

**Target**: `Cython.Utils.normalise_float_repr`
**Severity**: High
**Bug Type**: Logic, Crash
**Date**: 2025-09-25

## Summary

The `normalise_float_repr` function has multiple critical bugs when handling scientific notation with negative exponents, producing values that differ by many orders of magnitude from the input and crashing with ValueError on negative floats.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Property-based test for Cython.Utils.normalise_float_repr"""

from hypothesis import given, strategies as st
from Cython.Utils import normalise_float_repr

@given(st.floats(allow_nan=False, allow_infinity=False))
def test_normalise_float_repr_value_preservation(f):
    """Test that normalise_float_repr preserves the float value."""
    float_str = str(f)
    result = normalise_float_repr(float_str)
    assert float(float_str) == float(result), f"Value not preserved: {float_str} -> {result}"

# Run the test
if __name__ == "__main__":
    test_normalise_float_repr_value_preservation()
```

<details>

<summary>
**Failing input**: `f=6.103515625e-05` (and others)
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/13/hypo.py", line 16, in <module>
  |     test_normalise_float_repr_value_preservation()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/13/hypo.py", line 8, in test_normalise_float_repr_value_preservation
  |     def test_normalise_float_repr_value_preservation(f):
  |                    ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/13/hypo.py", line 12, in test_normalise_float_repr_value_preservation
    |     assert float(float_str) == float(result), f"Value not preserved: {float_str} -> {result}"
    |                                ~~~~~^^^^^^^^
    | ValueError: could not convert string to float: '.000-1'
    | Falsifying example: test_normalise_float_repr_value_preservation(
    |     f=-1e-05,
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/13/hypo.py", line 12, in test_normalise_float_repr_value_preservation
    |     assert float(float_str) == float(result), f"Value not preserved: {float_str} -> {result}"
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    | AssertionError: Value not preserved: 1.3002371459020878e-15 -> 130.0000000000000002371459020878
    | Falsifying example: test_normalise_float_repr_value_preservation(
    |     f=1.3002371459020878e-15,
    | )
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Demonstrate the bug in Cython.Utils.normalise_float_repr"""

from Cython.Utils import normalise_float_repr

# The failing test case from the bug report
original = 6.103515625e-05
float_str = '6.103515625e-05'
result = normalise_float_repr(float_str)

print(f"Input string:      {float_str}")
print(f"Expected behavior: Normalized string that preserves the float value")
print(f"                   (e.g., .00006103515625 or similar)")
print()
print(f"Actual result:     {result}")
print()
print(f"Original value:    {original}")
print(f"Result value:      {float(result)}")
print(f"Difference:        {abs(original - float(result))}")
print()
print(f"Values are equal?  {float(float_str) == float(result)}")
print()
print("ERROR: Result is off by approximately 10 billion times!")
```

<details>

<summary>
Output shows massive value corruption
</summary>
```
Input string:      6.103515625e-05
Expected behavior: Normalized string that preserves the float value
                   (e.g., .00006103515625 or similar)

Actual result:     610351.00005625

Original value:    6.103515625e-05
Result value:      610351.00005625
Difference:        610350.9999952149

Values are equal?  False

ERROR: Result is off by approximately 10 billion times!
```
</details>

## Why This Is A Bug

This violates the fundamental contract of `normalise_float_repr` as documented by its own test suite. In `/home/npc/miniconda/lib/python3.13/site-packages/Cython/Tests/TestCythonUtils.py:198`, the test explicitly verifies:

```python
self.assertEqual(float(float_str), float(result))
```

This requirement is clear: the normalized representation must preserve the float value when converted back. The function fails this requirement catastrophically in multiple ways:

1. **Magnitude Error**: For positive floats like `6.103515625e-05`, the result is off by a factor of ~10 billion. Instead of producing `.00006103515625`, it returns `610351.00005625`.

2. **ValueError Crash**: For negative floats like `-1e-05`, the function produces invalid output `.000-1` which cannot be parsed as a float, causing a ValueError.

3. **Systematic Logic Flaw**: The bug occurs because lines 667-677 in Utils.py incorrectly handle negative exponents when combined with decimal points. When `exp` is negative after adjustment (line 677: `exp += num_int_digits`), the slicing operation `str_value[:exp]` with negative indices doesn't behave as intended - it takes characters from the end instead of properly handling the decimal positioning.

This function is used in `Cython/Compiler/ExprNodes.py` for compile-time float constant handling where correctness is critical for code generation.

## Relevant Context

The function's purpose according to the docstring is to "Generate a 'normalised', simple digits string representation of a float value to allow string comparisons." However, the implementation fundamentally breaks when:

1. The input has scientific notation with a negative exponent (e.g., `e-05`)
2. The mantissa contains a decimal point (e.g., `6.103515625`)
3. The adjusted exponent after accounting for decimal position becomes negative

The algorithm at lines 679-685 attempts to construct the result using:
```python
result = (
    str_value[:exp]  # This breaks when exp is negative
    + '0' * (exp - len(str_value))
    + '.'
    + '0' * -exp
    + str_value[exp:]
).rstrip('0')
```

When `exp` is negative (e.g., -4), `str_value[:exp]` doesn't insert leading zeros as intended but instead takes characters from the beginning up to the 4th-to-last character, completely misplacing the decimal point.

## Proposed Fix

The fix requires properly handling negative exponents to correctly position the decimal point:

```diff
--- a/Cython/Utils.py
+++ b/Cython/Utils.py
@@ -677,14 +677,24 @@ def normalise_float_repr(float_str):
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
+        # Need leading zeros after decimal point
+        result = '.' + '0' * -exp + str_value
+    elif exp >= len(str_value):
+        # All digits before decimal, may need trailing zeros
+        result = str_value + '0' * (exp - len(str_value)) + '.'
+    else:
+        # Decimal point goes inside the digits
+        result = str_value[:exp] + '.' + str_value[exp:]
+
+    result = result.rstrip('0')
+
+    # Handle edge cases
+    if result == '.':
+        return '.0'
+    if result.endswith('.'):
+        return result + '0'

-    return result if result != '.' else '.0'
+    return result
```

This properly handles all three cases: negative exponents requiring leading zeros, positive exponents requiring trailing zeros, and exponents that place the decimal within the digit string.