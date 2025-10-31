# Bug Report: numpy.lib.scimath.power Returns NaN Imaginary Part for Small Negative Base

**Target**: `numpy.lib.scimath.power`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.lib.scimath.power()` returns a complex number with NaN imaginary part when given a very small negative base and a negative even integer power, instead of returning a proper complex number with 0 or finite imaginary part like it does for other negative bases.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume, settings
import numpy as np
import numpy.lib.scimath as scimath

@given(
    st.floats(min_value=-1e10, max_value=1e10, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=500)
def test_scimath_power_general(x, p):
    assume(abs(x) > 1e-100)
    assume(abs(p) > 1e-10 and abs(p) < 100)

    result = scimath.power(x, p)

    assert not np.isnan(result).any() if hasattr(result, '__iter__') else not np.isnan(result), \
        f"power({x}, {p}) should not be NaN"
```

**Failing input**: `x = -2.0758172915594093e-87, p = -4.0`

## Reproducing the Bug

```python
import numpy as np
import numpy.lib.scimath as scimath

x = -2.0758172915594093e-87
p = -4.0

result = scimath.power(x, p)
print(f"scimath.power({x}, {p}) = {result}")
print(f"Result: {result}")
print(f"Imaginary part: {result.imag}")
print(f"Imaginary part is NaN: {np.isnan(result.imag)}")

print("\nComparison with other negative bases:")
for test_x in [-1.0, -1e-10, -1e-50]:
    test_result = scimath.power(test_x, -4.0)
    print(f"scimath.power({test_x}, -4.0) = {test_result}")
```

Output:
```
scimath.power(-2.0758172915594093e-87, -4.0) = (inf+nanj)
Result: (inf+nanj)
Imaginary part: nan
Imaginary part is NaN: True

Comparison with other negative bases:
scimath.power(-1.0, -4.0) = (1+0j)
scimath.power(-1e-10, -4.0) = (9.999999999999999e+39+0j)
scimath.power(-1e-50, -4.0) = (9.999999999999999e+199+0j)
```

## Why This Is A Bug

1. **Inconsistency**: For all other negative bases with negative even integer powers, `scimath.power()` returns a complex number with imaginary part `0j`. For this extreme case, it returns `nanj`.

2. **Violated contract**: The docstring states "If `x` contains negative values, the output is converted to the complex domain." A complex number with NaN imaginary part is not a proper complex domain value.

3. **Expected behavior**: Since `p = -4.0` is a negative even integer, `x^(-4) = 1/(x^4)`. For any negative `x` and even power, the result should be a positive real number (or complex with 0 imaginary part when converted to complex domain). The result should be `(inf+0j)`, not `(inf+nanj)`.

4. **Comparison with numpy.power**: `numpy.power(-2.0758172915594093e-87, -4.0)` correctly returns `inf` (no NaN).

## Fix

The issue occurs when `x` is very small and negative, causing overflow to infinity when raised to a negative power. The conversion to complex domain introduces NaN in the imaginary part due to how the intermediate calculation handles the sign.

The fix should ensure that when the result overflows to infinity for negative bases with even integer powers, the imaginary part remains 0 (not NaN). This likely requires special handling in the `power` function or its helper functions (`_fix_real_lt_zero`, `_fix_int_lt_zero`) to detect overflow cases and ensure the imaginary part is set correctly.