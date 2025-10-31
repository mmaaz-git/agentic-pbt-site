# Bug Report: xarray.corr Returns Values Outside [-1, 1] Due to Floating-Point Precision

**Target**: `xarray.computation.computation.corr`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `xarray.corr()` function can return correlation values slightly outside the mathematically valid range of [-1, 1] due to floating-point precision errors. This violates the fundamental mathematical property that correlation coefficients must lie within [-1, 1].

## Property-Based Test

```python
from hypothesis import given, settings, assume, strategies as st
from hypothesis.extra.numpy import arrays
import numpy as np
import xarray as xr
from xarray.computation.computation import corr

@given(
    shape=st.tuples(st.integers(min_value=2, max_value=10), st.integers(min_value=2, max_value=10)),
    data_a=st.data(),
    data_b=st.data(),
)
@settings(max_examples=200)
def test_corr_bounded(shape, data_a, data_b):
    arr_a = data_a.draw(
        arrays(
            dtype=np.float64,
            shape=shape,
            elements=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        )
    )
    arr_b = data_b.draw(
        arrays(
            dtype=np.float64,
            shape=shape,
            elements=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        )
    )

    da_a = xr.DataArray(arr_a, dims=["x", "y"])
    da_b = xr.DataArray(arr_b, dims=["x", "y"])

    assume(da_a.std().item() > 1e-10)
    assume(da_b.std().item() > 1e-10)

    result = corr(da_a, da_b)

    assert np.all(result.values >= -1.0) and np.all(result.values <= 1.0)
```

**Failing input**:
```python
shape = (2, 8)
arr_a = np.array([[65., 65.,  0.,  0.,  0.,  0.,  0.,  0.],
                  [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])
arr_b = np.array([[65., 65.,  0.,  0.,  0.,  0.,  0.,  0.],
                  [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])
```

## Reproducing the Bug

```python
import numpy as np
import xarray as xr
from xarray.computation.computation import corr

data_a = np.array([[65., 65.,  0.,  0.,  0.,  0.,  0.,  0.],
                   [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])
data_b = np.array([[65., 65.,  0.,  0.,  0.,  0.,  0.,  0.],
                   [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])

da_a = xr.DataArray(data_a, dims=["x", "y"])
da_b = xr.DataArray(data_b, dims=["x", "y"])

result = corr(da_a, da_b)
print(f"Correlation: {result.values}")
print(f"In valid range [-1, 1]? {-1.0 <= result.values <= 1.0}")
```

Output:
```
Correlation: 1.0000000000000002
In valid range [-1, 1]? False
```

## Why This Is A Bug

1. **Mathematical violation**: Correlation coefficients are mathematically bounded to the range [-1, 1]. Any value outside this range is incorrect.

2. **Inconsistent with other libraries**: Both NumPy's `corrcoef` and pandas' `Series.corr` properly handle floating-point precision and return results within [-1, 1] for the same input:
   - `numpy.corrcoef`: Returns `1.0` (correct)
   - `pandas.Series.corr`: Returns `1.0` (correct)
   - `xarray.corr`: Returns `1.0000000000000002` (incorrect)

3. **Breaks downstream code**: Code that relies on the mathematical property `|corr| <= 1` may fail or produce incorrect results (e.g., computing `arccos(corr)` would raise an error).

4. **Not an extreme edge case**: This occurs with straightforward data patterns, not just pathological inputs.

## Fix

The fix is to clamp the correlation result to the valid range [-1, 1] after computation. This is a common practice in statistical libraries to handle floating-point precision issues.

```diff
--- a/xarray/computation/computation.py
+++ b/xarray/computation/computation.py
@@ -309,7 +309,10 @@ def _cov_corr(
             da_a_std = da_a.std(dim=dim)
             da_b_std = da_b.std(dim=dim)
         corr = cov / (da_a_std * da_b_std)
-        return cast(T_DataArray, corr)
+        # Clamp to [-1, 1] to handle floating-point precision issues
+        import xarray as xr
+        corr_clamped = xr.where(corr > 1, 1, xr.where(corr < -1, -1, corr))
+        return cast(T_DataArray, corr_clamped)
```

Alternatively, using numpy operations:

```diff
--- a/xarray/computation/computation.py
+++ b/xarray/computation/computation.py
@@ -309,7 +309,8 @@ def _cov_corr(
             da_a_std = da_a.std(dim=dim)
             da_b_std = da_b.std(dim=dim)
         corr = cov / (da_a_std * da_b_std)
-        return cast(T_DataArray, corr)
+        # Clamp to [-1, 1] to handle floating-point precision issues
+        return cast(T_DataArray, corr.clip(-1, 1))
```