# Bug Report: scipy.stats.norm.moment() Returns NaN for Small Location Parameters

**Target**: `scipy.stats.norm.moment()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`scipy.stats.norm.moment(1)` returns NaN instead of the correct value when the location parameter is extremely small (near machine epsilon), while `mean()` correctly returns the location parameter.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
import scipy.stats as stats

@settings(max_examples=100)
@given(
    st.floats(min_value=-5, max_value=5, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.5, max_value=3, allow_nan=False, allow_infinity=False),
)
def test_mean_moment_consistency(loc, scale):
    dist = stats.norm(loc=loc, scale=scale)
    mean_value = dist.mean()
    moment_value = dist.moment(1)

    assert np.isclose(mean_value, moment_value, rtol=1e-12), \
        f"mean() = {mean_value}, moment(1) = {moment_value}"
```

**Failing input**: `loc=2.225073858507e-311, scale=1.0`

## Reproducing the Bug

```python
import numpy as np
import scipy.stats as stats

loc = 2.225073858507e-311
scale = 1.0

dist = stats.norm(loc=loc, scale=scale)
mean_value = dist.mean()
moment_value = dist.moment(1)

print(f"mean() = {mean_value}")
print(f"moment(1) = {moment_value}")
```

**Output:**
```
mean() = 2.225073858507e-311
moment(1) = nan

RuntimeWarning: overflow encountered in divide
  fac = scale / loc
RuntimeWarning: invalid value encountered in multiply
  res2 += fac**n * val
```

## Why This Is A Bug

The first moment of any probability distribution must equal its mean by definition: E[X] = E[X^1]. For a normal distribution with location parameter `loc` and scale parameter `scale`, the first moment is exactly `loc`.

The bug occurs in scipy's moment calculation code (in `_distn_infrastructure.py`):

```python
fac = scale / loc
res2 += fac**n * val
```

When `loc` is extremely small (near machine epsilon), dividing `scale / loc` causes overflow, resulting in NaN propagation. This affects valid distribution parameters that scipy accepts without error.

The mean() method correctly returns `loc`, but moment(1) returns NaN, violating the fundamental property that `moment(1)` should equal `mean()`.

## Fix

The moment calculation should handle edge cases where `loc` is near zero. For the first moment specifically, it can directly return the mean without going through the generic moment calculation that involves division by `loc`.

```diff
--- a/scipy/stats/_distn_infrastructure.py
+++ b/scipy/stats/_distn_infrastructure.py
@@ -1350,6 +1350,9 @@ class rv_continuous(rv_generic):
         if not np.isfinite(loc):
             return np.nan

+        if order == 1:
+            return self.mean()
+
         fac = scale / loc
         res2 += fac**n * val
```

Alternatively, the division-based calculation should be protected against extreme values of `loc` to avoid overflow.