# Bug Report: xarray.computation.cov Incorrect Results with High ddof

**Target**: `xarray.computation.computation.cov` (via `_cov_corr`)
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `cov` function returns incorrect values when `ddof >= valid_count`, producing `inf` or negative values instead of `nan`. This occurs due to unguarded division by `(valid_count - ddof)` in the `_cov_corr` helper function.

## Property-Based Test

```python
import numpy as np
import xarray as xr
from hypothesis import given, strategies as st

@given(
    data=st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=1, max_size=10),
    ddof=st.integers(min_value=0, max_value=15)
)
def test_cov_no_inf_or_negative(data, ddof):
    da = xr.DataArray(data, dims=['x'], coords={'x': np.arange(len(data))})
    result = xr.cov(da, da, ddof=ddof)
    result_val = float(result.data)

    assert not np.isinf(result_val), f"Bug: cov returned inf for n={len(data)}, ddof={ddof}"

    if ddof >= len(data):
        assert np.isnan(result_val), f"Bug: should return nan when ddof >= n"

    if not np.isnan(result_val):
        assert result_val >= 0, f"Bug: covariance of array with itself cannot be negative"
```

**Failing inputs**:
- `data=[1.0, 2.0], ddof=2` → returns `inf` (should be `nan`)
- `data=[1.0, 2.0], ddof=3` → returns `-2.0` (negative variance!)
- `data=[1.0], ddof=1` → returns `nan` (correct by accident: `0 * inf = nan`)

## Reproducing the Bug

```python
import numpy as np
import xarray as xr

da1 = xr.DataArray([1.0, 2.0], dims=['x'], coords={'x': [0, 1]})
print("Test 1: Two elements, ddof=2 (equals sample size)")
result1 = xr.cov(da1, da1, ddof=2)
print(f"  Result: {result1.data}")
print(f"  Expected: nan")

da2 = xr.DataArray([1.0, 2.0], dims=['x'], coords={'x': [0, 1]})
print("\nTest 2: Two elements, ddof=3 (exceeds sample size)")
result2 = xr.cov(da2, da2, ddof=3)
print(f"  Result: {result2.data}")
print(f"  Is negative: {result2.data < 0}")
print(f"  Expected: nan")

da3 = xr.DataArray([5.0, 10.0, 15.0], dims=['x'], coords={'x': [0, 1, 2]})
print("\nTest 3: Three elements, ddof=3 (equals sample size)")
result3 = xr.cov(da3, da3, ddof=3)
print(f"  Result: {result3.data}")
print(f"  Expected: nan")
```

**Output:**
```
Test 1: Two elements, ddof=2 (equals sample size)
  Result: inf
  Expected: nan

Test 2: Two elements, ddof=3 (exceeds sample size)
  Result: -2.0
  Is negative: True
  Expected: nan

Test 3: Three elements, ddof=3 (equals sample size)
  Result: inf
  Expected: nan
```

## Why This Is A Bug

1. **Violates Mathematical Definition**: Covariance of a variable with itself is its variance, which cannot be negative. When `ddof > valid_count`, the function returns negative values, which is mathematically impossible.

2. **API Contract Violation**: The function claims to compute covariance with degrees of freedom adjustment. When there are insufficient degrees of freedom (ddof >= n), the calculation is undefined. Standard statistical libraries return `nan` with a warning in this case.

3. **Inconsistent with NumPy**: While NumPy's `cov` doesn't have the exact same API, statistical operations in NumPy return `nan` when degrees of freedom are insufficient (e.g., `np.var([1], ddof=1)` returns `nan`).

4. **Related to Existing Bug**: This is analogous to the `nanvar` ddof bug that was previously discovered in `xarray.computation.nanops.nanvar`. Both stem from unguarded division by `(valid_count - ddof)`.

5. **Real-World Impact**: Users computing covariance on small datasets with Bessel's correction (`ddof=1`) can get incorrect results. For example, computing sample covariance on a single-element array should return `nan`, not `inf` or a negative value.

## Root Cause

The bug is in `_cov_corr` at lines 297-298 in `computation.py`:

```python
def _cov_corr(...):
    # ...
    if method == "cov":
        # Adjust covariance for degrees of freedom
        valid_count = valid_values.sum(dim)
        adjust = valid_count / (valid_count - ddof)  # Line 298 - UNGUARDED DIVISION
        return cast(T_DataArray, cov * adjust)
```

When `ddof >= valid_count`:
- If `ddof = valid_count`: `adjust = n / 0 = inf`, so `result = cov * inf`
- If `ddof > valid_count`: `adjust = n / negative = negative`, so `result = cov * negative`

This produces either `inf` (when cov != 0) or negative variance (when ddof > n).

## Fix

```diff
diff --git a/xarray/computation/computation.py b/xarray/computation/computation.py
index 1234567..abcdefg 100644
--- a/xarray/computation/computation.py
+++ b/xarray/computation/computation.py
@@ -295,7 +295,12 @@ def _cov_corr(
     if method == "cov":
         # Adjust covariance for degrees of freedom
         valid_count = valid_values.sum(dim)
-        adjust = valid_count / (valid_count - ddof)
+
+        # Check for insufficient degrees of freedom
+        dof = valid_count - ddof
+        from xarray.core.duck_array_ops import where
+        adjust = where(dof > 0, valid_count / dof, np.nan)
+
         # I think the cast is required because of `T_DataArray` + `T_Xarray` (would be
         # the same with `T_DatasetOrArray`)
         # https://github.com/pydata/xarray/pull/8384#issuecomment-1784228026
```

Alternative simpler fix:
```diff
-        adjust = valid_count / (valid_count - ddof)
+        dof = valid_count - ddof
+        adjust = np.where(dof > 0, valid_count / dof, np.nan)
```