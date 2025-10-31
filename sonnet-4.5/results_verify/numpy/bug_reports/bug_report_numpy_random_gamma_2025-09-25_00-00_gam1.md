# Bug Report: numpy.random.gamma Produces Zero Values

**Target**: `numpy.random.Generator.gamma`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The gamma distribution implementation produces exact zero values with very small shape parameters, violating the mathematical requirement that the gamma distribution has support (0, ∞).

## Property-Based Test

```python
import numpy as np
from hypothesis import given, settings, strategies as st


@given(shape=st.floats(min_value=1e-10, max_value=1e-3))
@settings(max_examples=100)
def test_gamma_always_positive(shape):
    rng = np.random.Generator(np.random.PCG64(42))
    result = rng.gamma(shape, size=100)
    assert np.all(result > 0), f"gamma({shape}) produced zeros or negative values: min={np.min(result)}"
```

**Failing input**: `shape=0.0009791878680225864`

## Reproducing the Bug

```python
import numpy as np

rng = np.random.Generator(np.random.PCG64(42))
result = rng.gamma(0.001, size=100)

print(f"Zeros: {np.sum(result == 0)}/100")
print(f"Min value: {np.min(result):.6e}")
```

Output:
```
Zeros: 55/100
Min value: 0.000000e+00
```

## Why This Is A Bug

The gamma distribution has support (0, ∞), meaning P(X = 0) = 0. The distribution should never produce exact zero values. This appears to be a floating-point underflow issue when the shape parameter is very small (< 0.001). When shape is small, the distribution is heavily skewed toward small positive values, and the implementation appears to underflow to zero instead of producing very small positive numbers.

## Fix

The bug likely stems from floating-point underflow in the gamma sampling algorithm. The implementation should either:
1. Use higher precision arithmetic for extreme parameter values
2. Clamp results to a minimum positive value (e.g., smallest positive float64)
3. Document the valid parameter range and raise errors for parameters that cause numerical issues

Without access to the C implementation source, a specific patch cannot be provided.