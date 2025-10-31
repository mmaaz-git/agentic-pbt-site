# Bug Report: pandas.core.array_algos.quantile int8 dtype

**Target**: `pandas.core.array_algos.quantile.quantile_compat`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `quantile_compat` function produces incorrect and non-monotonic quantile values for int8 arrays due to NumPy's `percentile` function mishandling int8 dtype. This results in quantile values that violate the fundamental property that quantiles should be monotonically increasing.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings
from hypothesis.extra import numpy as npst
import pandas.core.array_algos.quantile as quantile_module


@given(
    values=npst.arrays(
        dtype=npst.integer_dtypes(endianness='='),
        shape=npst.array_shapes(min_dims=1, max_dims=1, min_side=2, max_side=100),
    ),
)
@settings(max_examples=300)
def test_quantile_integer_array(values):
    qs = np.array([0.0, 0.5, 1.0])
    interpolation = 'linear'

    result = quantile_module.quantile_compat(values, qs, interpolation)

    assert len(result) == len(qs)
    assert result[0] <= result[1] <= result[2]
```

**Failing input**: `array([-1, 127], dtype=int8)`

## Reproducing the Bug

```python
import numpy as np
import pandas.core.array_algos.quantile as quantile_module

values = np.array([-1, 127], dtype=np.int8)
qs = np.array([0.0, 0.5, 1.0])

result = quantile_module.quantile_compat(values, qs, 'linear')

print(f"Input: {values}")
print(f"Result: {result}")
print(f"Expected: [-1.0, 63.0, 127.0]")
print(f"Actual: {result}")
print(f"Monotonic: {result[0] <= result[1] <= result[2]}")
```

Output:
```
Input: [ -1 127]
Result: [ -1. 191. 127.]
Expected: [-1.0, 63.0, 127.0]
Actual: [ -1. 191. 127.]
Monotonic: False
```

## Why This Is A Bug

Quantiles must be monotonically increasing by definition: Q(p1) ≤ Q(p2) for p1 ≤ p2. This bug violates that fundamental property, returning Q(0.0) = -1.0, Q(0.5) = 191.0, Q(1.0) = 127.0, where the median (191.0) is larger than the maximum (127.0).

The root cause is that NumPy 2.3.0's `percentile` function gives incorrect results for int8 arrays, apparently treating negative int8 values as if they were unsigned. The value -1 in int8 has the same bit pattern as 255 in uint8, and (255 + 127) / 2 = 191.

While this is ultimately a NumPy bug, pandas should work around it to ensure correct behavior for users with int8 data.

## Fix

The fix is to upcast int8 (and potentially other small integer dtypes like uint8, int16, uint16) to a larger dtype (e.g., int64) before calling NumPy's percentile function. This is similar to how the code already handles datetime/timedelta types.

```diff
--- a/quantile.py
+++ b/quantile.py
@@ -176,6 +176,13 @@ def _nanpercentile(
     quantiles : scalar or array
     """

+    if values.dtype in [np.int8, np.uint8, np.int16, np.uint16]:
+        # Work around NumPy bug where percentile gives incorrect results
+        # for small integer dtypes
+        result = _nanpercentile(
+            values.astype(np.int64), qs=qs, na_value=na_value, mask=mask, interpolation=interpolation
+        )
+        return result
+
     if values.dtype.kind in "mM":
         # need to cast to integer to avoid rounding errors in numpy
         result = _nanpercentile(
```