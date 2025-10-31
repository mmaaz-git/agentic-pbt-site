# Bug Report: numpy.lcm Silent Integer Overflow

**Target**: `numpy.lcm`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.lcm` silently overflows when computing the LCM of large integers, returning negative values instead of raising an error or using a larger dtype.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, settings, strategies as st


@settings(max_examples=1000)
@given(st.integers(min_value=1, max_value=10**10), st.integers(min_value=1, max_value=10**10))
def test_gcd_lcm_product(a, b):
    gcd_val = np.gcd(a, b)
    lcm_val = np.lcm(a, b)
    product = gcd_val * lcm_val
    expected = abs(a * b)
    assert product == expected
```

**Failing input**: `a=3036988439, b=3037012561`

## Reproducing the Bug

```python
import numpy as np

a = 3_036_988_439
b = 3_037_012_561

lcm_val = np.lcm(a, b)
print(f"np.lcm({a}, {b}) = {lcm_val}")

assert lcm_val > 0, f"LCM of positive integers should be positive, got {lcm_val}"
```

Output:
```
np.lcm(3036988439, 3037012561) = -9223372036854769337
AssertionError: LCM of positive integers should be positive, got -9223372036854769337
```

## Why This Is A Bug

The documentation for `numpy.lcm` states: "Returns the lowest common multiple of |x1| and |x2|". The LCM of two positive integers is mathematically always positive. However, when the result exceeds `np.iinfo(np.int64).max`, the function silently overflows and returns a negative value instead of:
1. Raising an overflow error
2. Automatically promoting to a larger dtype (like Python's arbitrary precision integers)
3. Documenting this limitation

The correct LCM is `9,223,372,036,854,782,279`, but numpy returns `-9,223,372,036,854,769,337`.

## Fix

The function should either:

1. **Raise an overflow error** (preferred for correctness):
```diff
--- a/numpy/core/numeric.py
+++ b/numpy/core/numeric.py
@@ lcm function
+    # Check for overflow before returning
+    if result < 0 and x1 >= 0 and x2 >= 0:
+        raise OverflowError(f"LCM of {x1} and {x2} exceeds int64 range")
+    return result
```

2. **Promote to int64/object dtype automatically** when overflow would occur

3. **Document the limitation** clearly in the docstring that int64 overflow can occur and users should check results or use Python's `math.lcm` for large values