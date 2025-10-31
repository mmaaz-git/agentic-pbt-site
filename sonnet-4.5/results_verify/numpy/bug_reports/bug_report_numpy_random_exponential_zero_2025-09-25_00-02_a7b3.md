# Bug Report: numpy.random.exponential Zero Scale Parameter

**Target**: `numpy.random.Generator.exponential`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `exponential` distribution accepts `scale=0` without validation and returns 0.0, despite the exponential distribution being mathematically undefined at scale=0. The function validates negative scales but inconsistently allows zero.

## Property-Based Test

```python
import numpy.random as npr
from hypothesis import given, strategies as st
import pytest


@given(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False))
def test_exponential_scale_validation(scale):
    rng = npr.default_rng(42)

    if scale <= 0:
        with pytest.raises(ValueError):
            rng.exponential(scale)
    else:
        result = rng.exponential(scale)
        assert result >= 0
```

**Failing input**: `scale=0.0`

## Reproducing the Bug

```python
import numpy.random as npr

rng = npr.default_rng(42)

result = rng.exponential(0.0)
print(f"Result with scale=0: {result}")

try:
    result_neg = rng.exponential(-1.0)
except ValueError as e:
    print(f"Negative scale correctly raises: {e}")
```

**Output:**
```
Result with scale=0: 0.0
Negative scale correctly raises: scale < 0
```

## Why This Is A Bug

The exponential distribution has PDF f(x) = (1/β) * exp(-x/β) for x > 0, where β is the scale parameter. When β = 0, this involves division by zero and is mathematically undefined.

The function validates negative scales (raises ValueError for scale < 0) but inconsistently allows exactly zero, returning a degenerate distribution that always outputs 0.0.

**Affected distributions with same issue:**
- `gamma(shape=0.0, scale=1.0)` → returns 0.0
- `gamma(shape=1.0, scale=0.0)` → returns 0.0
- `rayleigh(scale=0.0)` → returns 0.0

Note: `beta` and `chisquare` correctly reject zero parameters, making this inconsistent across the numpy.random API.

## Fix

```diff
--- a/numpy/random/_generator.pyx
+++ b/numpy/random/_generator.pyx
@@ -exponential_method
-    if scale < 0:
-        raise ValueError("scale < 0")
+    if scale <= 0:
+        raise ValueError("scale must be positive")
```

Similar validation changes needed for `gamma` (shape and scale) and `rayleigh` (scale).