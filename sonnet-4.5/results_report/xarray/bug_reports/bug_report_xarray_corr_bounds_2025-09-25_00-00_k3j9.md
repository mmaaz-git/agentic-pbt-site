# Bug Report: xarray.computation.corr Correlation Coefficient Exceeds Valid Range

**Target**: `xarray.computation.computation.corr`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `xarray.corr()` function can return correlation coefficients that exceed the mathematically valid range of [-1, 1] due to floating-point arithmetic errors. Specifically, values like 1.00000000000000022 (greater than 1.0) can be returned, violating the fundamental mathematical property that correlation coefficients must be bounded.

## Property-Based Test

```python
from hypothesis import given, settings, assume, strategies as st
from hypothesis.extra.numpy import arrays
import numpy as np
import xarray as xr

@given(
    data=arrays(
        dtype=np.float64,
        shape=st.tuples(st.integers(2, 10), st.integers(2, 10)),
        elements=st.floats(
            min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
        ),
    )
)
@settings(max_examples=200)
def test_corr_bounds(data):
    da_a = xr.DataArray(data[:, 0], dims=["x"])
    da_b = xr.DataArray(data[:, 1], dims=["x"])

    assume(da_a.std().values > 1e-10)
    assume(da_b.std().values > 1e-10)

    correlation = xr.corr(da_a, da_b)
    corr_val = correlation.values.item() if correlation.values.ndim == 0 else correlation.values

    assert -1.0 <= corr_val <= 1.0
```

**Failing input**:
```python
data=array([[1., 1.],
           [0., 0.],
           [0., 0.]])
```

## Reproducing the Bug

```python
import numpy as np
import xarray as xr

data = np.array([[1., 1.],
                  [0., 0.],
                  [0., 0.]])

da_a = xr.DataArray(data[:, 0], dims=["x"])
da_b = xr.DataArray(data[:, 1], dims=["x"])

correlation = xr.corr(da_a, da_b)
corr_val = correlation.values.item()

print(f"Correlation value: {corr_val:.17f}")
print(f"Exceeds 1.0: {corr_val > 1.0}")

np_corr = np.corrcoef(data[:, 0], data[:, 1])[0, 1]
print(f"NumPy correlation: {np_corr:.17f}")
print(f"NumPy exceeds 1.0: {np_corr > 1.0}")
```

Output:
```
Correlation value: 1.00000000000000022
Exceeds 1.0: True
NumPy correlation: 1.00000000000000000
NumPy exceeds 1.0: False
```

## Why This Is A Bug

The correlation coefficient is mathematically guaranteed to be in the range [-1, 1]. The xarray implementation computes correlation as `cov / (std_a * std_b)`, which due to floating-point rounding errors can produce values slightly outside this range.

This violates the documented behavior and mathematical definition of correlation. NumPy's `corrcoef` function handles this correctly by clamping the result to the valid range, as shown in the reproducer above.

While the deviation is small (2.2e-16), it causes:
1. Violations of the correlation coefficient's mathematical properties
2. Unexpected behavior in downstream code that assumes valid bounds
3. Failed assertions in user code checking for valid correlations

## Fix

The fix is to clamp the correlation result to the valid range [-1, 1]. This can be done by modifying the `_cov_corr` function in `xarray/computation/computation.py`:

```diff
--- a/xarray/computation/computation.py
+++ b/xarray/computation/computation.py
@@ -309,7 +309,8 @@ def _cov_corr(
         else:
             da_a_std = da_a.std(dim=dim)
             da_b_std = da_b.std(dim=dim)
-        corr = cov / (da_a_std * da_b_std)
+        corr_raw = cov / (da_a_std * da_b_std)
+        corr = corr_raw.clip(min=-1.0, max=1.0)
         return cast(T_DataArray, corr)
```

This ensures that correlation values are always within the mathematically valid range, consistent with NumPy's behavior and the mathematical definition of correlation.