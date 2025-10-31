# Bug Report: numpy.random.weibull Produces Zero and Infinite Values

**Target**: `numpy.random.Generator.weibull`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The Weibull distribution implementation produces both exact zero values and infinite values with very small shape parameters, violating the mathematical requirement that the Weibull distribution has support (0, ∞) and should produce finite positive values.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, settings, strategies as st


@given(a=st.floats(min_value=1e-10, max_value=1e-3))
@settings(max_examples=50)
def test_weibull_always_positive(a):
    rng = np.random.Generator(np.random.PCG64(42))
    result = rng.weibull(a, size=100)
    assert np.all(result > 0), f"weibull({a}) produced zeros: min={np.min(result)}"
```

**Failing input**: `a=0.001`

## Reproducing the Bug

```python
import numpy as np

rng = np.random.Generator(np.random.PCG64(42))
result = rng.weibull(0.001, size=100)

print(f"Zeros: {np.sum(result == 0)}/100")
print(f"Infinities: {np.sum(np.isinf(result))}/100")
print(f"Min: {np.min(result[np.isfinite(result)])}")
```

Output:
```
Zeros: 46/100
Infinities: 10/100
Min: 0.0
```

## Why This Is A Bug

The Weibull distribution has support (0, ∞) and should only produce finite positive values. The implementation produces both zeros (likely from underflow) and infinities (likely from overflow) when the shape parameter is very small. This violates the mathematical properties of the distribution and could cause downstream errors in applications that assume valid Weibull samples are always finite and positive.

## Fix

The bug stems from numerical instability in the Weibull sampling algorithm with extreme shape parameters. The implementation should either:
1. Use more numerically stable algorithms for extreme parameter values
2. Validate and clamp results to ensure they are finite and positive
3. Document valid parameter ranges and raise errors for numerically unstable parameters

Without access to the C implementation source, a specific patch cannot be provided.