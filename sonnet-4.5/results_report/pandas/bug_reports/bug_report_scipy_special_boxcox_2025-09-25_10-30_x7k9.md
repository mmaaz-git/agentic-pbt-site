# Bug Report: scipy.special.boxcox Numerical Precision

**Target**: `scipy.special.boxcox` / `scipy.special.inv_boxcox`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The round-trip property `inv_boxcox(boxcox(x, lmbda), lmbda) == x` fails with significant numerical errors when `lmbda` is very negative (e.g., < -6) and `x` is moderately large (e.g., > 20). In some cases, the relative error exceeds 1%, and in extreme cases, `inv_boxcox` returns infinity or NaN.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
import numpy as np
from scipy import special

@given(
    st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False)
)
def test_boxcox_inv_boxcox_roundtrip(lmbda, x):
    y = special.boxcox(x, lmbda)
    assume(not np.isinf(y) and not np.isnan(y))
    x_recovered = special.inv_boxcox(y, lmbda)
    assert np.isclose(x, x_recovered, rtol=1e-6, atol=1e-6)
```

**Failing input**: `lmbda=-10, x=26`

## Reproducing the Bug

```python
import numpy as np
from scipy import special

lmbda = -10
x = 26

y = special.boxcox(x, lmbda)
x_recovered = special.inv_boxcox(y, lmbda)

print(f"Original x: {x}")
print(f"Recovered x: {x_recovered}")
print(f"Relative error: {abs(x - x_recovered) / x:.6f}")
```

Output:
```
Original x: 26
Recovered x: 25.992076683399546
Relative error: 0.000305
```

Additional failing cases:
- `lmbda=-10, x=50`: `inv_boxcox` returns `inf`
- `lmbda=-10, x=100`: `inv_boxcox` returns `inf`
- `lmbda=-8, x=100`: Relative error = 1.3%

## Why This Is A Bug

1. **Documentation implies round-trips should work**: The docstring for `inv_boxcox` provides an example showing a successful round-trip with no warnings about parameter limitations.

2. **No warnings for problematic parameter ranges**: The functions don't warn users when operating in parameter ranges where numerical stability is poor.

3. **Silent failures**: In some cases, `inv_boxcox` returns infinity without raising an error or warning, silently producing incorrect results.

4. **Inconsistent with documented behavior**: The documentation states what `inv_boxcox` should compute, but the actual results deviate significantly from the expected values in these cases.

## Fix

The root cause is numerical instability in the formula `(x**lmbda - 1) / lmbda` when `lmbda` is very negative and `x` is large, as `x**lmbda` becomes extremely small (close to 0), causing catastrophic cancellation.

### Option 1: Use a more stable formulation

For negative lambda values with large x, consider using logarithmic arithmetic or alternative formulations that avoid the subtraction of nearly-equal numbers.

### Option 2: Add parameter validation

Add warnings or errors when parameter combinations are known to be numerically unstable:

```python
if lmbda < -6 and x > 10:
    warnings.warn("Numerical instability may occur with very negative lambda and large x")
```

### Option 3: Document limitations

Update the docstring to explicitly warn about numerical stability issues:

```python
"""
...
Notes
-----
Numerical stability may be poor when lmbda is very negative (e.g., < -6)
and x is large (e.g., > 10). Round-trip accuracy is not guaranteed in these cases.
```