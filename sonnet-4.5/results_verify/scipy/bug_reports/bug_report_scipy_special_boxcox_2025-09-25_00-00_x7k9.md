# Bug Report: scipy.special boxcox/inv_boxcox Round-Trip Failure

**Target**: `scipy.special.boxcox` and `scipy.special.inv_boxcox`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The Box-Cox transformation functions `boxcox` and `inv_boxcox` fail to correctly round-trip when lambda is extremely small (at the subnormal float level, around 5e-324). While `boxcox` correctly treats such values as effectively zero and returns `log(x)`, `inv_boxcox` does not apply the same threshold logic, resulting in incorrect results.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
import scipy.special as sp
import math

@given(
    st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-2, max_value=2, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=1000)
def test_boxcox_inv_boxcox_roundtrip(x, lmbda):
    y = sp.boxcox(x, lmbda)
    x_reconstructed = sp.inv_boxcox(y, lmbda)
    assert math.isclose(x, x_reconstructed, rel_tol=1e-9, abs_tol=1e-9)
```

**Failing input**: `x=2.0, lmbda=5e-324`

## Reproducing the Bug

```python
import scipy.special as sp

x = 2.0
lmbda = 5e-324

y = sp.boxcox(x, lmbda)
x_reconstructed = sp.inv_boxcox(y, lmbda)

print(f"x = {x}")
print(f"lambda = {lmbda}")
print(f"boxcox(x, lambda) = {y}")
print(f"inv_boxcox(boxcox(x, lambda), lambda) = {x_reconstructed}")
print(f"Expected: {x}")
print(f"Actual error: {abs(x - x_reconstructed):.6f}")
```

Output:
```
x = 2.0
lambda = 5e-324
boxcox(x, lambda) = 0.6931471805599453
inv_boxcox(boxcox(x, lambda), lambda) = 2.718281828459045
Expected: 2.0
Actual error: 0.718282
```

The bug also affects the inverse direction:
```python
y = 0.5
lmbda = 5e-324
x = sp.inv_boxcox(y, lmbda)
y_reconstructed = sp.boxcox(x, lmbda)
```
Output:
```
inv_boxcox(0.5, 5e-324) = 1.0
boxcox(1.0, 5e-324) = 0.0
Expected: 0.5
```

## Why This Is A Bug

The Box-Cox transformation is defined as:
- `y = (x**λ - 1) / λ` if λ ≠ 0
- `y = log(x)` if λ = 0

And its inverse:
- `x = (λ*y + 1)**(1/λ)` if λ ≠ 0
- `x = exp(y)` if λ = 0

When lambda is extremely small (below ~1e-300), the formula `(x**λ - 1) / λ` should approach `log(x)` via L'Hôpital's rule. The `boxcox` function correctly recognizes this and returns `log(x)`. However, `inv_boxcox` does not apply consistent threshold logic and attempts to compute `(λ*y + 1)**(1/λ)`, which produces incorrect results due to catastrophic loss of precision.

For the round-trip property to hold, both functions must use the same threshold for when to apply the λ=0 formulas.

## Fix

The issue is that `boxcox` and `inv_boxcox` use inconsistent thresholds for determining when lambda is "close enough to zero" to use the limit formulas. The fix would be to:

1. Define a consistent threshold (e.g., `|λ| < 1e-200`)
2. Apply this threshold in both `boxcox` and `inv_boxcox`
3. When `|λ|` is below the threshold, use the λ=0 formulas:
   - `boxcox`: return `log(x)`
   - `inv_boxcox`: return `exp(y)`

Since these are likely implemented in C as ufuncs, the actual implementation would need to be modified in the scipy C code, but the high-level fix is to ensure both functions use the same threshold for switching to the limit case.