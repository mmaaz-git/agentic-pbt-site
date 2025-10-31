# Bug Report: Cython.Utils.normalise_float_repr - Catastrophic Numeric Corruption for Scientific Notation

**Target**: `Cython.Utils.normalise_float_repr`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`normalise_float_repr` produces catastrophically wrong numeric values for floating-point numbers in scientific notation with negative exponents, with errors ranging from 10^4 to 10^16 times the actual value.

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
**Failing input**: `5.960464477539063e-08`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/45/hypo.py", line 13, in <module>
    test_normalise_float_repr_value_preservation()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/45/hypo.py", line 6, in test_normalise_float_repr_value_preservation
    @given(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e100, max_value=1e100))
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/45/hypo.py", line 10, in test_normalise_float_repr_value_preservation
    assert math.isclose(float(normalized), float(float_str), rel_tol=1e-15)
           ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_normalise_float_repr_value_preservation(
    x=5.960464477539063e-08,
)
```
</details>

## Reproducing the Bug

```python
from Cython.Utils import normalise_float_repr

x = "6.103515625e-05"
result = normalise_float_repr(x)
print(f"Input:  {x}")
print(f"Output: {result}")
print(f"Input value:  {float(x)}")
print(f"Output value: {float(result)}")
print(f"Expected value: {6.103515625e-05}")
print(f"Error factor: {float(result) / float(x)}")
```

<details>

<summary>
Output shows value corrupted by factor of ~10^10
</summary>
```
Input:  6.103515625e-05
Output: 610351.00005625
Input value:  6.103515625e-05
Output value: 610351.00005625
Expected value: 6.103515625e-05
Error factor: 9999990784.9216
```
</details>

## Why This Is A Bug

This violates the explicit contract established by the test suite at `/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Tests/TestCythonUtils.py:198` which requires: `self.assertEqual(float(float_str), float(result))` - the function must preserve numeric values.

The bug occurs specifically when scientific notation numbers have:
- Negative exponents
- Mantissa decimal places exceeding `|exponent| - num_int_digits`

The algorithmic flaw is in lines 672-677 of Utils.py. After removing the decimal point from the mantissa, the code incorrectly adjusts the exponent by adding `num_int_digits` (line 677), fundamentally misunderstanding how scientific notation exponents relate to decimal position after digit concatenation.

## Relevant Context

The bug manifests in a clear pattern based on mantissa decimal places:
- "1.234e-5" (4 decimals) → Works correctly
- "1.2345e-5" (5 decimals) → **FAILS** (off by 8.1e4)
- "1.23456e-5" (6 decimals) → **FAILS** (off by 9.7e5)
- "5.960464477539063e-08" (15 decimals) → **FAILS** (off by 1e16)

The existing test suite only covers mantissas with up to 3 decimal places, explaining why this severe bug went undetected. While `normalise_float_repr` is an internal utility function, Cython is a compiler that generates C code - incorrect float representation could lead to seriously wrong generated code.

Documentation: [Cython.Utils source](https://github.com/cython/cython/blob/master/Cython/Utils.py#L660-L687)
Test suite: [TestCythonUtils.py](https://github.com/cython/cython/blob/master/Cython/Tests/TestCythonUtils.py#L190-L202)

## Proposed Fix

The core issue is that the algorithm incorrectly calculates the final exponent position. When the decimal is removed from the mantissa, the code needs to account for how many fractional digits were present, not just add the integer digit count:

```diff
--- a/Cython/Utils.py
+++ b/Cython/Utils.py
@@ -671,11 +671,12 @@ def normalise_float_repr(float_str):

     if '.' in str_value:
         num_int_digits = str_value.index('.')
+        num_frac_digits = len(str_value) - num_int_digits - 1
         str_value = str_value[:num_int_digits] + str_value[num_int_digits + 1:]
+        exp = exp - num_frac_digits + num_int_digits
     else:
         num_int_digits = len(str_value)
-    exp += num_int_digits
-
+        exp += num_int_digits
+
     result = (
         str_value[:exp]
         + '0' * (exp - len(str_value))
```