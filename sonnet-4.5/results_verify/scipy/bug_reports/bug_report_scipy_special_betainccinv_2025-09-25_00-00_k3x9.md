# Bug Report: scipy.special.betainccinv Invalid Return Value

**Target**: `scipy.special.betainccinv`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`betainccinv` incorrectly returns `x=1.0` for certain inputs, violating the fundamental property that `betaincc(a, b, 1) = 0` for all valid parameters. This causes the round-trip property `betaincc(a, b, betainccinv(a, b, y)) = y` to fail completely for small y values.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/scipy_env')

import math
from hypothesis import given, strategies as st, settings
from scipy import special


@given(
    a=st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False),
    b=st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False),
    y=st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=1000)
def test_betaincc_betainccinv_roundtrip(a, b, y):
    x = special.betainccinv(a, b, y)

    if not (0 <= x <= 1):
        return

    result = special.betaincc(a, b, x)

    assert math.isclose(result, y, rel_tol=1e-7, abs_tol=1e-7), \
        f"betaincc({a}, {b}, betainccinv({a}, {b}, {y})) = {result}, expected {y}"
```

**Failing input**: `a=1.0, b=0.1, y=0.01`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/scipy_env')

from scipy import special

a, b, y = 1.0, 0.1, 0.01

x = special.betainccinv(a, b, y)
print(f"betainccinv({a}, {b}, {y}) = {x}")

result = special.betaincc(a, b, x)
print(f"betaincc({a}, {b}, {x}) = {result}")
print(f"Expected: {y}")
print(f"Error: {abs(result - y)}")
```

Output:
```
betainccinv(1.0, 0.1, 0.01) = 1.0
betaincc(1.0, 0.1, 1.0) = 0.0
Expected: 0.01
Error: 0.01
```

## Why This Is A Bug

The function `betainccinv(a, b, y)` is documented to compute x such that `betaincc(a, b, x) = y`. However:

1. Mathematically, `betaincc(a, b, 1) = 0` for all valid `a, b` (since it's the complement of the complete beta function)
2. Therefore, `betainccinv` should never return `x=1.0` for any `y > 0`
3. In the failing case, `betainccinv(1.0, 0.1, 0.01)` returns `1.0`, but `betaincc(1.0, 0.1, 1.0) = 0.0 ≠ 0.01`

This violates the documented inverse relationship and causes 100% relative error in the result.

## Fix

The issue appears to be in the boundary handling or convergence criteria of the inverse routine. The function should recognize that when `y > 0`, the solution `x` must be strictly less than `1.0`. A proper fix would involve:

1. Adding a check to ensure `x < 1.0` when `y > 0`
2. Adjusting the root-finding algorithm to avoid converging to the boundary `x=1.0` when it's not a valid solution
3. Potentially using higher precision near the boundary `x ≈ 1.0` where the function has steep gradients

Without access to the C++ implementation details (this wraps Boost's `ibetac_inv`), a precise patch cannot be provided. The fix would need to be made in the underlying Boost library or in scipy's wrapper code to add validation and handle edge cases appropriately.