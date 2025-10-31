# Bug Report: numpy.random.wald Produces Negative Values

**Target**: `numpy.random.Generator.wald`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The Wald (Inverse Gaussian) distribution implementation produces negative values with very small scale parameters, violating the mathematical requirement that the Wald distribution has support (0, ∞).

## Property-Based Test

```python
import numpy as np
from hypothesis import assume, given, settings, strategies as st


@given(
    mean=st.floats(allow_nan=False, allow_infinity=False, min_value=1e-10, max_value=1e2),
    scale=st.floats(min_value=1e-10, max_value=1e2),
)
@settings(max_examples=50)
def test_wald_always_positive(mean, scale):
    assume(mean > 0)
    assume(scale > 0)
    rng = np.random.Generator(np.random.PCG64(42))
    result = rng.wald(mean, scale, size=100)
    assert np.all(result > 0), f"wald({mean}, {scale}) produced zeros: min={np.min(result)}"
```

**Failing input**: `mean=1.0, scale=1e-10`

## Reproducing the Bug

```python
import numpy as np

rng = np.random.Generator(np.random.PCG64(42))
result = rng.wald(1.0, 1e-10, size=100)

print(f"Negative values: {np.sum(result < 0)}/100")
print(f"Min value: {np.min(result):.6e}")
```

Output:
```
Negative values: 60/100
Min value: -8.274037e-08
```

## Why This Is A Bug

The Wald distribution (also known as Inverse Gaussian) has support (0, ∞), meaning it should only produce strictly positive values. Producing negative values violates the mathematical definition of the distribution. This appears to be a numerical precision issue when the scale parameter is very small, possibly due to the implementation algorithm not handling extreme parameters correctly.

## Fix

The bug likely stems from numerical instability in the Wald distribution sampling algorithm when scale is very small. The implementation should either:
1. Add bounds checking to ensure all sampled values are positive (clamp to a small epsilon)
2. Use a more numerically stable algorithm for extreme parameter values
3. Raise a warning or error when parameters are outside the numerically stable range

Without access to the C implementation source, a specific patch cannot be provided, but the fix would involve modifying the underlying random generation code to handle extreme scale values more carefully.