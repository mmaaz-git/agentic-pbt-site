# Bug Report: scipy.special.inv_boxcox1p Returns Wrong Value for Extremely Small Lambda

**Target**: `scipy.special.inv_boxcox1p`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`scipy.special.inv_boxcox1p` returns incorrect values when lambda is extremely small (approximately < 1e-200) but non-zero. Instead of computing the inverse transformation, it incorrectly returns the input `y` unchanged, breaking the round-trip property with `boxcox1p`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume, settings
import scipy.special
import numpy as np
import math

@given(
    x=st.floats(min_value=-0.99, max_value=1e6, allow_nan=False, allow_infinity=False),
    lmbda=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=1000)
def test_boxcox1p_inv_boxcox1p_roundtrip(x, lmbda):
    y = scipy.special.boxcox1p(x, lmbda)
    assume(not np.isnan(y) and not np.isinf(y))
    result = scipy.special.inv_boxcox1p(y, lmbda)
    assert math.isclose(result, x, rel_tol=1e-9, abs_tol=1e-12)
```

**Failing input**: `x=1.0, lmbda=6.964131297701778e-264`

## Reproducing the Bug

```python
import scipy.special
import numpy as np

x = 1.0
lmbda = 1e-264

y = scipy.special.boxcox1p(x, lmbda)
print(f"boxcox1p({x}, {lmbda}) = {y}")

result = scipy.special.inv_boxcox1p(y, lmbda)
print(f"inv_boxcox1p({y}, {lmbda}) = {result}")
print(f"Expected: {x}")
print(f"Error: {abs(result - x)}")
```

Output:
```
boxcox1p(1.0, 1e-264) = 0.6931471805599453
inv_boxcox1p(0.6931471805599453, 1e-264) = 0.6931471805599453
Expected: 1.0
Error: 0.3068528194400547
```

The function returns `y` unchanged instead of computing the correct inverse.

## Why This Is A Bug

The Box-Cox transformation and its inverse should form a round-trip: `inv_boxcox1p(boxcox1p(x, λ), λ) = x`.

According to the docstring:
- `boxcox1p(x, λ) = ((1+x)^λ - 1) / λ` for λ ≠ 0, or `log(1+x)` for λ = 0
- `inv_boxcox1p(y, λ) = (y·λ + 1)^(1/λ) - 1` for λ ≠ 0, or `exp(y) - 1` for λ = 0

For very small λ, `boxcox1p` correctly uses the limiting form `log(1+x)`, so for x=1.0, it returns ln(2) ≈ 0.693.

However, `inv_boxcox1p` appears to have a threshold (around λ ≈ 1e-200) below which it incorrectly returns `y` unchanged instead of computing `exp(y) - 1 = exp(0.693) - 1 = 1.0`.

When λ = 0 exactly, the function works correctly:
```python
y = scipy.special.boxcox1p(1.0, 0.0)
result = scipy.special.inv_boxcox1p(y, 0.0)
```
Returns `result = 1.0` as expected.

The threshold behavior:
- λ = 1e-100: works correctly
- λ = 1e-200: returns wrong result (returns y instead of exp(y) - 1)
- λ = 1e-300: returns wrong result

## Fix

The bug appears to be in the threshold logic that decides when to use the λ=0 limiting case. When `inv_boxcox1p` detects very small λ and switches to the λ=0 formula, it should compute `exp(y) - 1`, but instead it's returning `y`.

The fix should ensure that when λ is below the threshold, the function computes:
```
x = exp(y) - 1
```

not simply:
```
x = y
```

This appears to be a simple implementation error where the wrong formula is being applied in the small-λ branch.