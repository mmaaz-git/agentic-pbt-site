# Bug Report: scipy.special.inv_boxcox1p Incorrect Inverse for Extremely Small Lambda

**Target**: `scipy.special.inv_boxcox1p`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`scipy.special.inv_boxcox1p` fails to correctly compute the inverse of `boxcox1p` when the lambda parameter is extremely small (approximately < 1e-200). Instead of returning the correct inverse value, the function incorrectly returns the input `y` unchanged.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import scipy.special as sp
import math

@given(
    st.floats(min_value=1e-10, max_value=1e6, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False)
)
def test_boxcox1p_inv_boxcox1p_roundtrip(x, lmbda):
    """Test that inv_boxcox1p(boxcox1p(x, lmbda), lmbda) = x"""
    y = sp.boxcox1p(x, lmbda)
    result = sp.inv_boxcox1p(y, lmbda)
    assert math.isclose(result, x, rel_tol=1e-7, abs_tol=1e-10), \
        f"inv_boxcox1p(boxcox1p({x}, {lmbda}), {lmbda}) = {result}, expected {x}"
```

**Failing input**: `x=1.0, lmbda=3.9731703875764937e-287`

## Reproducing the Bug

```python
import scipy.special as sp

x = 1.0
lmbda = 1e-300

y = sp.boxcox1p(x, lmbda)
result = sp.inv_boxcox1p(y, lmbda)

print(f"boxcox1p({x}, {lmbda}) = {y}")
print(f"inv_boxcox1p({y}, {lmbda}) = {result}")
print(f"Expected: {x}")
print(f"Actual: {result}")
print(f"Error: {abs(result - x)}")
```

Output:
```
boxcox1p(1.0, 1e-300) = 0.6931471805599453
inv_boxcox1p(0.6931471805599453, 1e-300) = 0.6931471805599453
Expected: 1.0
Actual: 0.6931471805599453
Error: 0.3068528194400547
```

## Why This Is A Bug

The function `inv_boxcox1p` is explicitly documented as the inverse of `boxcox1p`. The fundamental property of inverse functions is that `f^(-1)(f(x)) = x` for all valid inputs. This property is violated for lambda values smaller than approximately 1e-200.

When lambda is very small:
- `boxcox1p(1.0, 1e-300)` correctly computes and returns `log(1+1) ≈ 0.693`
- `inv_boxcox1p(0.693, 1e-300)` should return `1.0` but instead returns `0.693` unchanged

The bug does not occur when:
- lambda = 0.0 exactly (works correctly)
- lambda >= 1e-100 approximately (works correctly)

This suggests a numerical precision issue in the implementation when handling extremely small non-zero lambda values, possibly related to how the function decides whether to use the lambda=0 code path versus the lambda≠0 code path.

## Fix

This appears to be a numerical precision issue in the C implementation. The function likely needs to check if lambda is sufficiently close to zero (not just exactly zero) and use the appropriate formula:

When `|lmbda|` is very small (e.g., `< 1e-200`), the function should use:
```
x = exp(y) - 1
```

instead of the general formula:
```
x = (1 + y * lmbda)^(1/lmbda) - 1
```

The threshold for switching between these formulas needs to be adjusted to prevent numerical issues with extremely small lambda values.

A similar fix pattern can be seen in how `boxcox1p` itself handles the lambda=0 case - `inv_boxcox1p` needs a similar epsilon-based check rather than an exact zero check.