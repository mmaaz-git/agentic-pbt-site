# Bug Report: scipy.special.boxcox1p Inconsistent Lambda Handling

**Target**: `scipy.special.boxcox1p` and `scipy.special.inv_boxcox1p`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When lambda is very small (approximately < 1e-157), `boxcox1p` and `inv_boxcox1p` use inconsistent logic to decide whether to treat lambda as zero, violating the documented inverse relationship and causing round-trip failures with large errors (~0.31).

## Property-Based Test

```python
import math
from hypothesis import given, strategies as st, settings
from scipy.special import boxcox1p, inv_boxcox1p


@settings(max_examples=1000)
@given(
    y=st.floats(min_value=-1e10, max_value=1e10, allow_nan=False, allow_infinity=False),
    lmbda=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False)
)
def test_boxcox1p_round_trip_boxcox_first(y, lmbda):
    x = inv_boxcox1p(y, lmbda)
    if not math.isfinite(x) or x <= -1:
        return
    y_recovered = boxcox1p(x, lmbda)
    assert math.isfinite(y_recovered), f"boxcox1p returned non-finite value: {y_recovered}"
    assert math.isclose(y_recovered, y, rel_tol=1e-9, abs_tol=1e-9), \
        f"Round-trip failed: y={y}, lmbda={lmbda}, x={x}, y_recovered={y_recovered}"
```

**Failing input**: `y=1.0, lmbda=5.808166112732823e-234`

## Reproducing the Bug

```python
from scipy.special import boxcox1p, inv_boxcox1p

y = 1.0
lmbda = 5.808166112732823e-234

x = inv_boxcox1p(y, lmbda)
y_recovered = boxcox1p(x, lmbda)

print(f"Input: y={y}, lambda={lmbda}")
print(f"inv_boxcox1p(y, lambda) = {x}")
print(f"boxcox1p(x, lambda) = {y_recovered}")
print(f"Expected: {y}")
print(f"Actual: {y_recovered}")
print(f"Error: {abs(y_recovered - y)}")
```

Output:
```
Input: y=1.0, lambda=5.808166112732823e-234
inv_boxcox1p(y, lambda) = 1.0
boxcox1p(x, lambda) = 0.6931471805599453
Expected: 1.0
Actual: 0.6931471805599453
Error: 0.3068528194400547
```

The value 0.6931471805599453 equals `log(2)`, indicating that `boxcox1p` is using the `lambda==0` special case even though lambda is not exactly zero. However, `inv_boxcox1p` is using the general formula for non-zero lambda, creating an inconsistency.

## Why This Is A Bug

The documentation explicitly states that `inv_boxcox1p` computes the inverse of `boxcox1p`. For any valid input, the round-trip property should hold: `boxcox1p(inv_boxcox1p(y, lmbda), lmbda) ≈ y`.

The bug occurs because the two functions use different threshold values to decide when lambda is "close enough to zero" to use the special case `log(1+x)` formula:
- `boxcox1p` treats lambda < ~1e-157 as zero
- `inv_boxcox1p` uses a different (larger) threshold

This creates a range of lambda values where the two functions are inconsistent, violating the inverse relationship with errors as large as 0.31.

## Fix

The fix requires ensuring both `boxcox1p` and `inv_boxcox1p` use the same threshold for determining when lambda should be treated as zero. The functions should use a consistent epsilon value (e.g., comparing `abs(lambda) < epsilon`) to decide which formula to use.

Since these are likely implemented as C/Cython ufuncs in scipy.special, the fix would need to be applied in the C source code where both functions check if lambda is zero.