# Bug Report: xarray.computation.corr Correlation Coefficient Exceeds Valid Range

**Target**: `xarray.computation.computation.corr`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `xarray.corr()` function returns correlation coefficients that exceed the mathematically valid range of [-1, 1] due to floating-point arithmetic errors, violating fundamental mathematical properties of the Pearson correlation coefficient.

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

    assert -1.0 <= corr_val <= 1.0, f"Correlation {corr_val} is outside valid range [-1, 1]"

# Run the test
test_corr_bounds()
```

<details>

<summary>
**Failing input**: `array([[36.     ,  1.     ], [ 2.00001,  2.00001]])`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/1/hypo.py", line 29, in <module>
    test_corr_bounds()
    ~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/1/hypo.py", line 7, in test_corr_bounds
    data=arrays(
               ^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/1/hypo.py", line 26, in test_corr_bounds
    assert -1.0 <= corr_val <= 1.0, f"Correlation {corr_val} is outside valid range [-1, 1]"
           ^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Correlation -1.0000000000000002 is outside valid range [-1, 1]
Falsifying example: test_corr_bounds(
    data=array([[36.     ,  1.     ],
           [ 2.00001,  2.00001]]),
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
import xarray as xr

# Create the specific failing input from the bug report
data = np.array([[1., 1.],
                  [0., 0.],
                  [0., 0.]])

# Create DataArrays from the columns
da_a = xr.DataArray(data[:, 0], dims=["x"])
da_b = xr.DataArray(data[:, 1], dims=["x"])

# Print input data for clarity
print("Input data:")
print(f"da_a values: {da_a.values}")
print(f"da_b values: {da_b.values}")
print()

# Compute correlation using xarray
correlation = xr.corr(da_a, da_b)
corr_val = correlation.values.item()

# Display results
print("xarray correlation results:")
print(f"Correlation value: {corr_val:.17f}")
print(f"Exceeds 1.0: {corr_val > 1.0}")
print(f"Amount over 1.0: {corr_val - 1.0:.2e}")
print()

# Compare with NumPy's corrcoef
np_corr = np.corrcoef(data[:, 0], data[:, 1])[0, 1]
print("NumPy correlation results:")
print(f"Correlation value: {np_corr:.17f}")
print(f"Exceeds 1.0: {np_corr > 1.0}")
```

<details>

<summary>
xarray correlation exceeds valid range while NumPy correctly bounds it
</summary>
```
Input data:
da_a values: [1. 0. 0.]
da_b values: [1. 0. 0.]

xarray correlation results:
Correlation value: 1.00000000000000022
Exceeds 1.0: True
Amount over 1.0: 2.22e-16

NumPy correlation results:
Correlation value: 1.00000000000000000
Exceeds 1.0: False
```
</details>

## Why This Is A Bug

The Pearson correlation coefficient is mathematically guaranteed to lie within the range [-1, 1]. This is a fundamental property derived from the Cauchy-Schwarz inequality: |cov(X,Y)| ≤ σ_X * σ_Y, which implies |ρ| ≤ 1.

The xarray implementation violates this mathematical constraint due to floating-point rounding errors in the computation `corr = cov / (da_a_std * da_b_std)` at line 312 of `/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages/xarray/computation/computation.py`. When arrays are perfectly correlated or anti-correlated, small numerical errors in computing the covariance and standard deviations can cause the final division to slightly exceed the theoretical bounds.

While the deviation is small (typically around 1e-16), this violation causes real problems:
1. **Mathematical invariant violation**: Code that relies on correlation being in [-1, 1] may fail
2. **Failed assertions**: User code checking `assert -1 <= corr <= 1` will fail unexpectedly
3. **Inconsistency with NumPy**: NumPy's `corrcoef` correctly handles this case by clamping results

The xarray documentation states it computes "the Pearson correlation coefficient" which, by definition, must be in [-1, 1]. The function references `pandas.Series.corr` as corresponding functionality, implying similar behavior should be expected.

## Relevant Context

The issue occurs in the `_cov_corr` function which is the internal implementation for both `xr.cov()` and `xr.corr()`. The problematic code is at line 312:

```python
corr = cov / (da_a_std * da_b_std)
```

NumPy's implementation includes safeguards against this floating-point issue, ensuring results stay within valid bounds. Testing shows NumPy returns exactly 1.0 for the same inputs where xarray returns 1.00000000000000022.

Documentation: The xarray.corr function is documented at https://docs.xarray.dev/en/stable/generated/xarray.corr.html

## Proposed Fix

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