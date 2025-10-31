# Bug Report: Cython.Utils.normalise_float_repr Incorrect Handling of Scientific Notation

**Target**: `Cython.Utils.normalise_float_repr`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `normalise_float_repr` function produces incorrect results when normalizing floats in scientific notation, particularly for values with negative exponents. The function violates its fundamental property that the normalized string should represent the same numeric value as the input.

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
```

**Failing inputs**:
- `5.960464477539063e-08`
- `-3.0929648190816446e-178`

## Reproducing the Bug

```python
from Cython.Utils import normalise_float_repr

f1 = 5.960464477539063e-08
result1 = normalise_float_repr(str(f1))
print(f"Input: {f1}")
print(f"Result: {result1}")
print(f"Float of result: {float(result1)}")

f2 = -3.0929648190816446e-178
result2 = normalise_float_repr(str(f2))
print(f"Input: {f2}")
print(f"Result: {result2}")
```

Output:
```
Input: 5.960464477539063e-08
Result: 596046447.00000007539063
Float of result: 596046447.0000001
BUG: Expected 5.960464477539063e-08, got 596046447.0000001

Input: -3.0929648190816446e-178
Result: .00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000-30929648190816446
BUG: Cannot parse result as float
```

## Why This Is A Bug

The function claims to "generate a 'normalised', simple digits string representation of a float value to allow string comparisons" (line 662). The existing unit tests in `TestCythonUtils.py` verify that `float(float_str) == float(result)` for the normalized output (line 198).

This bug violates the core contract of the function:
1. For `5.960464477539063e-08`, it produces `596046447.0...` - a value that is ~10 billion times larger
2. For `-3.0929648190816446e-178`, it produces an unparseable string with the minus sign in the middle

The root cause appears to be incorrect handling of negative exponents in the normalization logic (lines 679-685).

## Fix

The bug is in the exponent handling logic. When dealing with negative exponents (small numbers like `1e-8`), the current code incorrectly calculates the position of the decimal point. A comprehensive fix requires rewriting the exponent normalization logic to correctly handle:

1. Negative exponents (numbers < 1)
2. Negative numbers (the minus sign should be preserved and handled separately)
3. Edge cases where `exp` is negative and needs to add leading zeros

The logic at lines 679-685 needs to be revised to properly place the decimal point for negative exponents.