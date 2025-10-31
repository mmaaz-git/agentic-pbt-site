# Bug Report: scipy.special.boxcox/inv_boxcox Subnormal Lambda Inconsistency

**Target**: `scipy.special.boxcox` and `scipy.special.inv_boxcox`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `boxcox` and `inv_boxcox` functions handle subnormal (extremely small) lambda values inconsistently, violating the documented inverse relationship and breaking the round-trip property.

## Property-Based Test

```python
import math
from hypothesis import given, strategies as st, settings
from scipy import special

@given(
    st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-2, max_value=2, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=1000)
def test_boxcox_inv_boxcox_round_trip(x, lmbda):
    y = special.boxcox(x, lmbda)
    result = special.inv_boxcox(y, lmbda)
    assert math.isclose(result, x, rel_tol=1e-9, abs_tol=1e-9)
```

**Failing input**: `x=0.5, lmbda=5e-324` (subnormal float)

## Reproducing the Bug

```python
from scipy import special
import numpy as np

x = 0.5
lmbda = 5e-324

y = special.boxcox(x, lmbda)
print(f"boxcox(0.5, 5e-324) = {y}")
print(f"Expected log(0.5) = {np.log(0.5)}")

result = special.inv_boxcox(y, lmbda)
print(f"inv_boxcox({y}, 5e-324) = {result}")
print(f"Expected: 0.5")
print(f"Actual: {result}")

print(f"\nWith exact zero:")
y_zero = special.boxcox(x, 0.0)
result_zero = special.inv_boxcox(y_zero, 0.0)
print(f"inv_boxcox(boxcox(0.5, 0.0), 0.0) = {result_zero}")
```

## Why This Is A Bug

According to the documentation, `boxcox` and `inv_boxcox` are inverse functions. The documentation explicitly states that `inv_boxcox` should find `x` such that `y = boxcox(x, lmbda)`.

For lambda = 5e-324 (a subnormal float that is not exactly 0 but extremely close):
- `boxcox(0.5, 5e-324)` returns -0.693147... (i.e., log(0.5)), using the lambda=0 special case
- `inv_boxcox(-0.693147..., 5e-324)` returns 0.367879... (approximately 1/e), NOT using the lambda=0 special case

This breaks the fundamental inverse property: `inv_boxcox(boxcox(x, lmbda), lmbda) != x`

The functions handle the lambda=0 special case inconsistently - `boxcox` treats very small lambda as zero, but `inv_boxcox` does not, leading to incorrect results.

## Fix

The fix should ensure both `boxcox` and `inv_boxcox` use the same threshold for determining when lambda should be treated as 0. Currently:

- `boxcox` appears to check if lambda is close to 0 (or uses a threshold)
- `inv_boxcox` appears to check only if lambda == 0 exactly

Both functions should use the same tolerance/threshold when deciding to use the lambda=0 special case. A reasonable approach would be to use a small epsilon (e.g., `abs(lmbda) < 1e-10`) or handle subnormal floats explicitly.