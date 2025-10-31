# Bug Report: numpy.lib.scimath.power NaN in Imaginary Part for Overflow Cases

**Target**: `numpy.lib.scimath.power`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When `scimath.power` raises a small negative number to a negative even power that causes overflow/underflow, it returns a complex number with NaN in the imaginary part (`inf+nanj`) instead of a valid result.

## Property-Based Test

```python
import numpy as np
import numpy.lib.scimath as scimath
from hypothesis import given, strategies as st

@given(
    st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
    st.integers(min_value=-5, max_value=5)
)
def test_power_definition(x, n):
    result = scimath.power(x, n)
    assert not np.isnan(result)
```

**Failing input**: `x = -9.499558537778752e-188, n = -2`

## Reproducing the Bug

```python
import numpy as np
import numpy.lib.scimath as scimath

x = -1e-200
n = -2
result = scimath.power(x, n)

print(f'scimath.power({x}, {n}) = {result}')
print(f'Has NaN: {np.isnan(result)}')
```

Output:
```
scimath.power(-1e-200, -2) = (inf+nanj)
Has NaN: True
```

Expected: `inf` or `inf+0j` (since `(-1e-200)^(-2) = 1/(1e-400) ≈ inf` and negative^even is positive)
Actual: `inf+nanj` (complex with NaN imaginary part)

## Why This Is A Bug

The `scimath.power` function is designed to handle negative bases by converting to the complex domain. However, when raising a negative number to an even power, the mathematical result is always real (positive).

In this case:
- `(-1e-200) ** (-2) = 1 / ((-1e-200) ** 2) = 1 / (1e-400) ≈ inf`
- The result should be real infinity or at worst `inf+0j`, not `inf+nanj`

The function's own documentation shows that even powers of negative numbers should have valid imaginary parts:
```python
>>> np.emath.power([-2, 4], 2)
array([ 4.-0.j, 16.+0.j])
```

The NaN arises because:
1. `scimath.power` converts `-1e-200` to complex `-1e-200+0j`
2. `(-1e-200+0j) ** 2` underflows to `-0j` (negative zero)
3. `1 / (-0j)` produces `inf+nanj` in numpy

## Fix

The fix should detect when the power is an even integer and avoid unnecessary complex conversion for negative bases in those cases. Here's a potential patch:

```diff
--- a/numpy/lib/_scimath_impl.py
+++ b/numpy/lib/_scimath_impl.py
@@ -488,6 +488,14 @@ def power(x, p):
     array([ 4, 256])

     """
+    # For even integer powers, negative bases produce real results
+    # Avoid complex conversion to prevent inf+nanj in overflow cases
+    p_arr = asarray(p)
+    if (p_arr.dtype.kind in 'iu' and  # integer type
+        nx.all(p_arr % 2 == 0)):      # even power
+        # Use regular power for even integer powers
+        return nx.power(x, p)
+
     x = _fix_real_lt_zero(x)
     p = _fix_int_lt_zero(p)
     return nx.power(x, p)
```

This ensures that for even integer powers, we skip the complex conversion and use regular `numpy.power`, which handles overflow correctly by returning `inf` instead of `inf+nanj`.