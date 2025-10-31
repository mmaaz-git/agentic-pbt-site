# Bug Report: scipy.special.inv_boxcox1p Fails Round-Trip for Subnormal Lambda

**Target**: `scipy.special.inv_boxcox1p`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`inv_boxcox1p` fails to correctly invert `boxcox1p` when the lambda parameter is subnormal (< 2.225e-308), violating the fundamental round-trip property.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import scipy.special as sp
import math

@given(
    st.floats(min_value=-1 + 1e-10, max_value=100, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-5, max_value=5, allow_nan=False, allow_infinity=False)
)
def test_boxcox1p_inv_boxcox1p_roundtrip(x, lmbda):
    y = sp.boxcox1p(x, lmbda)
    if not (np.isnan(y) or np.isinf(y)):
        result = sp.inv_boxcox1p(y, lmbda)
        assert math.isclose(result, x, rel_tol=1e-7, abs_tol=1e-9)
```

**Failing input**: `x=1.0, lmbda=1.1125369292536007e-308`

## Reproducing the Bug

```python
import numpy as np
import scipy.special as sp

x = 1.0
lmbda = 1e-308

y = sp.boxcox1p(x, lmbda)
x_recovered = sp.inv_boxcox1p(y, lmbda)

print(f"Input: x = {x}, lmbda = {lmbda}")
print(f"boxcox1p(x, lmbda) = {y}")
print(f"inv_boxcox1p(y, lmbda) = {x_recovered}")
print(f"Expected: {x}")
print(f"Error: {abs(x - x_recovered)}")
```

Output:
```
Input: x = 1.0, lmbda = 1e-308
boxcox1p(x, lmbda) = 0.6931471805599453
inv_boxcox1p(y, lmbda) = 0.6931471805599453
Expected: 1.0
Error: 0.3068528194400547
```

## Why This Is A Bug

The Box-Cox transformation and its inverse should satisfy the round-trip property: `inv_boxcox1p(boxcox1p(x, lmbda), lmbda) == x` for all valid inputs.

The issue occurs because:

1. `boxcox1p` correctly handles subnormal lambda values by treating them as `lmbda == 0`, computing `log(1 + x)`
2. `inv_boxcox1p` does NOT have this special handling, attempting to compute `(1 + lmbda * y)^(1/lmbda) - 1`
3. For subnormal lambda, `lmbda * y` underflows to 0, causing the formula to degenerate to `1^(huge_number) - 1 = 0`
4. This returns the transformed value `y` instead of the original value `x`

When `lmbda = 0` exactly, both functions work correctly. The bug only manifests for subnormal positive values.

## Fix

`inv_boxcox1p` should check for subnormal lambda values and handle them the same way as `lmbda == 0`:

```diff
The fix should be applied in the C/Cython implementation of inv_boxcox1p.
Add a check for very small |lmbda| values:

-    if (lmbda == 0) {
+    if (lmbda == 0 || fabs(lmbda) < DBL_MIN) {
         return expm1(y);
     } else {
         return pow(1 + lmbda * y, 1/lmbda) - 1;
     }

Where DBL_MIN is the smallest normal positive double-precision number (approximately 2.225e-308).
```