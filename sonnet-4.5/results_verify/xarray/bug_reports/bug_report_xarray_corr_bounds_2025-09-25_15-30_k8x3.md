# Bug Report: xarray.computation.corr - Correlation can exceed [-1, 1] bounds

**Target**: `xarray.computation.computation.corr`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `corr` function can return correlation values slightly outside the mathematical bounds of [-1, 1] due to floating point precision errors. This violates the fundamental mathematical property that Pearson correlation coefficients must lie in the interval [-1, 1].

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st, assume
from hypothesis.extra import numpy as npst
import numpy as np
import xarray as xr
from xarray.computation.computation import corr


@given(
    data=st.data(),
    shape=st.tuples(st.integers(2, 20), st.integers(2, 20))
)
@settings(max_examples=200)
def test_corr_bounded(data, shape):
    arr_a = data.draw(npst.arrays(
        dtype=np.float64,
        shape=shape,
        elements=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
    ))
    arr_b = data.draw(npst.arrays(
        dtype=np.float64,
        shape=shape,
        elements=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
    ))

    da_a = xr.DataArray(arr_a, dims=['x', 'y'])
    da_b = xr.DataArray(arr_b, dims=['x', 'y'])

    std_a = da_a.std().values
    std_b = da_b.std().values
    assume(std_a > 1e-10 and std_b > 1e-10)

    result = corr(da_a, da_b)
    val = result.values.item()

    assert -1.0 <= val <= 1.0, f"Correlation {val} exceeds bounds [-1, 1]"
```

**Failing input**: Arrays with shape `(2, 2)` where both are `[[0., 1.], [1., 1.]]`

## Reproducing the Bug

```python
import numpy as np
import xarray as xr
from xarray.computation.computation import corr

arr = np.array([[0., 1.], [1., 1.]])
da = xr.DataArray(arr, dims=['x', 'y'])

result = corr(da, da)
val = result.values.item()

print(f"corr(A, A) = {val}")
print(f"Expected: 1.0 (within bounds [-1, 1])")
print(f"Actual: {val}")
print(f"Exceeds upper bound: {val > 1.0}")
print(f"Difference from 1.0: {val - 1.0}")
```

Output:
```
corr(A, A) = 1.0000000000000002
Expected: 1.0 (within bounds [-1, 1])
Actual: 1.0000000000000002
Exceeds upper bound: True
Difference from 1.0: 2.220446049250313e-16
```

## Why This Is A Bug

The Pearson correlation coefficient is mathematically defined to always lie in the interval [-1, 1]. This is a fundamental property taught in every statistics course and relied upon by downstream code. While the violation is small (≈ 2e-16), it:

1. Breaks the mathematical invariant that correlation ∈ [-1, 1]
2. Can cause issues in code that checks `if corr > 1.0` or uses correlation values in assertions
3. Can propagate errors in downstream computations that assume valid correlation ranges

This is a well-known numerical issue in correlation computation. Other statistical libraries (like R's `cor()` and SciPy) handle this by clipping the result to [-1, 1].

## Fix

The fix is straightforward: clip the correlation result to ensure it stays within [-1, 1].

```diff
--- a/xarray/computation/computation.py
+++ b/xarray/computation/computation.py
@@ -310,7 +310,8 @@ def _cov_corr(
             da_a_std = da_a.std(dim=dim)
             da_b_std = da_b.std(dim=dim)
         corr = cov / (da_a_std * da_b_std)
-        return cast(T_DataArray, corr)
+        # Clip to [-1, 1] to handle floating point precision errors
+        return cast(T_DataArray, corr.clip(-1.0, 1.0))


 def cross(
```

This ensures that the correlation always satisfies the mathematical constraint -1 ≤ ρ ≤ 1, even in the presence of floating point rounding errors.