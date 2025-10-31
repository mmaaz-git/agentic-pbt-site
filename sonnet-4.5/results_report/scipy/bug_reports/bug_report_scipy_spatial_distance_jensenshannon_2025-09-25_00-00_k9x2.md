# Bug Report: scipy.spatial.distance.jensenshannon Invalid Base Parameter

**Target**: `scipy.spatial.distance.jensenshannon`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `jensenshannon` function does not validate the `base` parameter, allowing mathematically invalid values (base ≤ 0 or base = 1) that produce incorrect results (inf or nan) with only runtime warnings instead of proper exceptions.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings
from scipy.spatial.distance import jensenshannon


@settings(max_examples=500)
@given(
    st.integers(min_value=2, max_value=10),
    st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False)
)
def test_jensenshannon_base_produces_finite_result(k, base):
    x = np.random.rand(k) + 0.1
    y = np.random.rand(k) + 0.1
    x = x / x.sum()
    y = y / y.sum()

    result = jensenshannon(x, y, base=base)

    assert np.isfinite(result), \
        f"Jensen-Shannon should produce finite result for base={base}, got {result}"
```

**Failing input**: `base=1.0`

## Reproducing the Bug

```python
import numpy as np
from scipy.spatial.distance import jensenshannon

p = np.array([0.5, 0.5])
q = np.array([0.3, 0.7])

result = jensenshannon(p, q, base=1.0)
print(f"Result: {result}")
print(f"Is infinite: {np.isinf(result)}")
```

Expected: `ValueError` with message about invalid base
Actual: Returns `inf` with RuntimeWarning

## Why This Is A Bug

The function accepts invalid `base` parameters that are mathematically undefined:
- `base=1.0` causes division by zero since `log(1) = 0`, returning inf
- `base≤0` produces nan since logarithms are undefined for non-positive bases
- The function should raise a clear `ValueError` for invalid inputs rather than silently returning inf/nan

This violates the API contract of a distance function which should either:
1. Return a valid finite distance, or
2. Raise an explicit exception for invalid parameters

## Fix

```diff
--- a/scipy/spatial/distance.py
+++ b/scipy/spatial/distance.py
@@ -1370,6 +1370,9 @@ def jensenshannon(p, q, base=None, *, axis=0, keepdims=False):
     """
     p = np.asarray(p)
     q = np.asarray(q)
+    if base is not None and (base <= 0 or base == 1.0):
+        raise ValueError(f"base must be a positive number not equal to 1, got {base}")
+
     p = p / np.sum(p, axis=axis, keepdims=True)
     q = q / np.sum(q, axis=axis, keepdims=True)
     m = (p + q) / 2.0
```