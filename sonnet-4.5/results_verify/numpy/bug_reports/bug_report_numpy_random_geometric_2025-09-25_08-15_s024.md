# Bug Report: numpy.random.geometric Integer Overflow with Small Probabilities

**Target**: `numpy.random.geometric()`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.random.geometric()` returns the minimum int64 value (-9223372036854775808) instead of valid positive integers when the probability parameter `p` is very small (approximately p < 1e-100). This violates the fundamental contract of the geometric distribution, which must always return values >= 1.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
import numpy as np

@given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
def test_geometric_positive(p):
    assume(0 < p <= 1)
    samples = np.random.geometric(p, size=100)
    assert np.all(samples >= 1)
```

**Failing input**: `p=2.225073858507e-311`

## Reproducing the Bug

```python
import numpy as np

p = 1e-300
np.random.seed(42)
samples = np.random.geometric(p, size=10)
print(f"p = {p}")
print(f"Samples: {samples}")
print(f"Min value: {samples.min()}")
print(f"Expected: All values >= 1")
print(f"Actual: {samples.min()} (int64 minimum)")
```

Output:
```
p = 1e-300
Samples: [-9223372036854775808 -9223372036854775808 -9223372036854775808 ...]
Min value: -9223372036854775808
Expected: All values >= 1
Actual: -9223372036854775808 (int64 minimum)
```

## Why This Is A Bug

The geometric distribution models the number of trials until the first success in a sequence of Bernoulli trials. By definition, it must return positive integers >= 1.

When p is very small, the expected value is 1/p, which can exceed int64 range. However, instead of:
- Raising a clear error about parameter range limitations
- Returning infinity or a sentinel value
- Clamping to int64 max

The function silently overflows and returns int64 min (-9223372036854775808), which is mathematically nonsensical and could corrupt downstream computations.

The bug affects p values roughly below 1e-100. The new `Generator.geometric()` API has a similar overflow issue but returns int64 max instead.

## Fix

The function should validate the input parameter and raise an informative error when p is too small to represent the result in int64:

```diff
--- a/numpy/random/mtrand.pyx
+++ b/numpy/random/mtrand.pyx
@@ geometric function implementation
+    if p < 1e-18:
+        raise ValueError(
+            f"p={p} is too small. The expected number of trials (1/p) "
+            f"would overflow int64. Use p >= 1e-18 or consider using "
+            f"a different approach for extremely rare events."
+        )
```

Alternatively, the function could return float64 instead of int64 when the expected value exceeds int64 range, or clamp results to int64 max with a warning.