# Bug Report: numpy.random.dirichlet Zero Alpha Parameters

**Target**: `numpy.random.Generator.dirichlet`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `dirichlet` distribution accepts zero or negative alpha values without validation and produces mathematically invalid output. When all alpha values are zero, it returns an array that sums to 0 instead of 1, violating the fundamental simplex constraint of the Dirichlet distribution.

## Property-Based Test

```python
import numpy as np
import numpy.random as npr
from hypothesis import given, strategies as st


@given(st.integers(min_value=2, max_value=10))
def test_dirichlet_all_zeros_violates_simplex_constraint(size):
    rng = npr.default_rng(42)

    alpha = np.zeros(size)
    result = rng.dirichlet(alpha)

    assert np.isclose(result.sum(), 1.0)
```

**Failing input**: `alpha = [0.0, 0.0]` (or any array of zeros)

## Reproducing the Bug

```python
import numpy as np
import numpy.random as npr

rng = npr.default_rng(42)

alpha_zeros = [0.0, 0.0, 0.0]
result = rng.dirichlet(alpha_zeros)

print(f"Alpha: {alpha_zeros}")
print(f"Result: {result}")
print(f"Sum: {result.sum()}")
```

**Output:**
```
Alpha: [0.0, 0.0, 0.0]
Result: [0. 0. 0.]
Sum: 0.0
```

## Why This Is A Bug

The Dirichlet distribution is mathematically defined only for alpha_i > 0 for all i. The outputs must lie on the (n-1)-simplex, meaning they must sum to exactly 1.0.

When given zero alpha values, `dirichlet()`:
1. Produces output that sums to 0.0 instead of 1.0 (violates simplex constraint)
2. Accepts mathematically invalid parameters without error
3. Behaves inconsistently with the related `beta` distribution, which correctly rejects zero parameters with ValueError

This silently produces incorrect results that could corrupt statistical analyses or simulations.

## Fix

```diff
--- a/numpy/random/_generator.pyx
+++ b/numpy/random/_generator.pyx
@@ -dirichlet_implementation
+    if np.any(alpha <= 0):
+        raise ValueError("alpha values must be positive")
     # existing implementation
```

The validation should match the behavior of `beta`, which already correctly rejects non-positive parameters.