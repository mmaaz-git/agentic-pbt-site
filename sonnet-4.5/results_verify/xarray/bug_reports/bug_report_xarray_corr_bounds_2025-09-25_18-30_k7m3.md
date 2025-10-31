# Bug Report: xarray.corr Correlation Bounds Violation

**Target**: `xarray.computation.corr`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `xarray.corr()` function can return correlation values slightly outside the mathematically required bounds of [-1, 1] due to floating-point precision issues in the calculation. The Pearson correlation coefficient is mathematically defined to always be in [-1, 1], but xarray's implementation does not enforce this invariant.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume, settings
from hypothesis.extra.numpy import arrays
import numpy as np
import xarray as xr
from xarray import DataArray


@settings(max_examples=500)
@given(
    data=st.data(),
    size=st.integers(min_value=2, max_value=10),
)
def test_corr_bounds(data, size):
    data_a = data.draw(arrays(
        dtype=np.float64,
        shape=(size,),
        elements=st.floats(
            min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
        ),
    ))
    data_b = data.draw(arrays(
        dtype=np.float64,
        shape=(size,),
        elements=st.floats(
            min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
        ),
    ))
    assume(np.std(data_a) > 1e-10)
    assume(np.std(data_b) > 1e-10)

    da_a = DataArray(data_a, dims=["time"])
    da_b = DataArray(data_b, dims=["time"])

    corr_ab = xr.corr(da_a, da_b)

    assert -1 <= corr_ab.values <= 1, \
        f"Correlation {corr_ab.values} outside [-1, 1] bounds"
```

**Failing input**: `data_a = array([0., 15., 15., 15., 15.])`, `data_b = array([1., 1.5, 1.5, 1.5, 1.5])`

## Reproducing the Bug

```python
import numpy as np
import xarray as xr
from xarray import DataArray

data_a = np.array([0., 15., 15., 15., 15.])
data_b = np.array([1., 1.5, 1.5, 1.5, 1.5])

da_a = DataArray(data_a, dims=["time"])
da_b = DataArray(data_b, dims=["time"])

corr_result = xr.corr(da_a, da_b)

print(f"Correlation: {corr_result.values:.17f}")
print(f"Expected: value in [-1, 1]")
print(f"Actual: {corr_result.values:.17f} (exceeds 1 by {(corr_result.values - 1):.2e})")
print(f"Bounds check: {-1 <= corr_result.values <= 1}")

np_corr = np.corrcoef(data_a, data_b)[0, 1]
print(f"\nNumPy's corrcoef: {np_corr:.17f}")
print(f"NumPy within bounds: {-1 <= np_corr <= 1}")
```

Output:
```
Correlation: 1.00000000000000044
Expected: value in [-1, 1]
Actual: 1.00000000000000044 (exceeds 1 by 4.4e-16)
Bounds check: False

NumPy's corrcoef: 1.00000000000000000
NumPy within bounds: True
```

## Why This Is A Bug

1. **Mathematical Definition**: The Pearson correlation coefficient is mathematically defined to always be in the range [-1, 1]. This is a fundamental property of the statistic.

2. **Documentation**: The function's docstring states it computes "the Pearson correlation coefficient between two DataArray objects", implying it should satisfy all mathematical properties of that coefficient.

3. **Comparison with NumPy**: NumPy's `corrcoef` function returns exactly 1.0 for the same input, demonstrating that this issue can be avoided.

4. **Practical Impact**: User code that checks correlation bounds (e.g., `if corr > 1`) will fail unexpectedly. Downstream calculations that assume the mathematical property holds may produce incorrect results.

## Fix

The fix is to clip the correlation result to ensure it stays within the mathematically valid bounds of [-1, 1]:

```diff
--- a/xarray/computation/computation.py
+++ b/xarray/computation/computation.py
@@ -309,7 +309,8 @@ def _cov_corr(
         else:
             da_a_std = da_a.std(dim=dim)
             da_b_std = da_b.std(dim=dim)
-        corr = cov / (da_a_std * da_b_std)
+        corr_raw = cov / (da_a_std * da_b_std)
+        corr = corr_raw.clip(min=-1, max=1)
         return cast(T_DataArray, corr)
```

This ensures that floating-point precision errors don't cause the result to violate the mathematical bounds of the Pearson correlation coefficient, matching the behavior of NumPy and other statistical libraries.