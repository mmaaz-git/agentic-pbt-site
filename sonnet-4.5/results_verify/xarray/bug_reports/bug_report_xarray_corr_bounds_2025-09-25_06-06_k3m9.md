# Bug Report: xarray.corr Exceeds Mathematical Bounds

**Target**: `xarray.corr`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `xarray.corr()` function can return correlation values that exceed the mathematical bounds of [-1, 1] due to floating-point arithmetic errors. This violates the fundamental mathematical property that Pearson correlation coefficients must lie within [-1, 1].

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, assume
from hypothesis.extra.numpy import arrays
import numpy as np
import xarray as xr

@given(
    data=arrays(
        dtype=np.float64,
        shape=st.tuples(st.integers(2, 10), st.integers(2, 10)),
        elements=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    )
)
@settings(max_examples=500)
def test_corr_range(data):
    assume(np.std(data[:, 0]) > 1e-10)
    assume(np.std(data[:, 1]) > 1e-10)

    da_a = xr.DataArray(data[:, 0], dims=["x"])
    da_b = xr.DataArray(data[:, 1], dims=["x"])

    corr = xr.corr(da_a, da_b)

    assert -1.0 <= corr.values <= 1.0
```

**Failing input**:
```python
data=array([[1.9375, 1.    ],
           [0.    , 0.    ],
           [0.    , 0.    ],
           [0.    , 0.    ],
           [0.    , 0.    ]])
```

## Reproducing the Bug

```python
import numpy as np
import xarray as xr

data = np.array([[1.9375, 1.    ],
                 [0.    , 0.    ],
                 [0.    , 0.    ],
                 [0.    , 0.    ],
                 [0.    , 0.    ]])

da_a = xr.DataArray(data[:, 0], dims=["x"])
da_b = xr.DataArray(data[:, 1], dims=["x"])

corr = xr.corr(da_a, da_b)

print(f"Correlation: {corr.values.item()}")
print(f"Exceeds bounds: {corr.values.item() > 1.0}")

numpy_corr = np.corrcoef(data[:, 0], data[:, 1])[0, 1]
print(f"NumPy correlation: {numpy_corr}")
print(f"NumPy within bounds: {-1.0 <= numpy_corr <= 1.0}")
```

Output:
```
Correlation: 1.0000000000000002
Exceeds bounds: True
NumPy correlation: 0.9999999999999996
NumPy within bounds: True
```

## Why This Is A Bug

The Pearson correlation coefficient has a well-defined mathematical range of [-1, 1]. This is a fundamental mathematical property, not just a convention. Values outside this range are mathematically impossible and indicate numerical errors in the computation.

While floating-point rounding errors are expected, the result should be clipped to the valid range to maintain the mathematical guarantee. NumPy's `corrcoef` stays within bounds, but xarray's implementation exceeds 1.0 by approximately machine epsilon (~2.22e-16).

This bug can affect:
1. Code that relies on the mathematical bounds for validation
2. Algorithms that use correlation values in further computations (e.g., division, arccos)
3. Statistical tests that assume correlations are within [-1, 1]

## Fix

The fix is to clip the correlation result to the valid range [-1, 1] after computation, similar to what other correlation implementations do:

```diff
--- a/xarray/computation/computation.py
+++ b/xarray/computation/computation.py
@@ -310,7 +310,7 @@ def _cov_corr(
             da_a_std = da_a.std(dim=dim)
             da_b_std = da_b.std(dim=dim)
         corr = cov / (da_a_std * da_b_std)
-        return cast(T_DataArray, corr)
+        return cast(T_DataArray, corr.clip(-1.0, 1.0))
```

This ensures that floating-point rounding errors never cause the correlation to exceed the mathematical bounds of [-1, 1].