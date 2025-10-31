# Bug Report: SparseArray Aggregation Methods Incorrectly Handle NaN Values

**Target**: `pandas.core.arrays.sparse.SparseArray`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `sum()` and `mean()` methods of `SparseArray` incorrectly exclude NaN values from their calculations, returning finite values instead of propagating NaN as expected. This violates NumPy semantics where NaN should propagate through aggregation operations.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume, settings
from hypothesis.extra.numpy import arrays as st_arrays
import pandas.core.arrays.sparse as sparse
import numpy as np


@st.composite
def sparse_array_data(draw):
    dtype = draw(st.sampled_from([np.int64, np.float64, bool]))
    size = draw(st.integers(min_value=1, max_value=100))

    if dtype == np.int64:
        fill_value = draw(st.sampled_from([0, 1, -1]))
        arr = draw(st_arrays(dtype=dtype, shape=size))
    elif dtype == np.float64:
        fill_value = draw(st.sampled_from([0.0, 1.0, np.nan]))
        arr = draw(st_arrays(dtype=dtype, shape=size,
                             elements=st.floats(allow_nan=False, allow_infinity=False,
                                               min_value=-1e6, max_value=1e6) | st.just(np.nan)))
    else:
        fill_value = draw(st.booleans())
        arr = draw(st_arrays(dtype=dtype, shape=size))

    return arr, fill_value


@given(sparse_array_data())
@settings(max_examples=200)
def test_sum_matches_dense(data):
    """sum() on sparse should match sum() on dense"""
    arr, fill_value = data
    assume(not np.isnan(fill_value))

    sparse_arr = sparse.SparseArray(arr, fill_value=fill_value)
    dense = sparse_arr.to_dense()

    sparse_sum = sparse_arr.sum()
    dense_sum = dense.sum()

    if np.isnan(sparse_sum) and np.isnan(dense_sum):
        return

    assert sparse_sum == dense_sum or np.isclose(sparse_sum, dense_sum), \
        f"sum() mismatch: sparse={sparse_sum}, dense={dense_sum}"
```

**Failing input**: `data=(array([nan]), 0.0)`

## Reproducing the Bug

```python
import pandas.core.arrays.sparse as sparse
import numpy as np

arr = np.array([np.nan])
sparse_arr = sparse.SparseArray(arr, fill_value=0.0)
dense_arr = sparse_arr.to_dense()

print(f"Dense sum: {dense_arr.sum()}")
print(f"Sparse sum: {sparse_arr.sum()}")

arr2 = np.array([1.0, np.nan, 2.0])
sparse_arr2 = sparse.SparseArray(arr2, fill_value=0.0)
dense_arr2 = sparse_arr2.to_dense()

print(f"\nDense sum: {dense_arr2.sum()}")
print(f"Sparse sum: {sparse_arr2.sum()}")

print(f"\nDense mean: {dense_arr2.mean()}")
print(f"Sparse mean: {sparse_arr2.mean()}")
```

Expected output:
```
Dense sum: nan
Sparse sum: nan
```

Actual output:
```
Dense sum: nan
Sparse sum: 0.0

Dense sum: nan
Sparse sum: 3.0

Dense mean: nan
Sparse mean: 1.5
```

## Why This Is A Bug

NumPy aggregation operations propagate NaN values by design - if any value in the array is NaN (and skipna=True is not set), the result should be NaN. This is a fundamental property that users rely on for detecting missing or invalid data in their calculations.

The SparseArray implementation violates this contract because it uses `_valid_sp_values` which incorrectly filters out NaN values. While NaN values are correctly stored in the sparse array (as evidenced by `to_dense()` working correctly), the aggregation methods exclude them from calculations.

This affects:
- `sum()` - returns sum of non-NaN values instead of NaN
- `mean()` - returns mean of non-NaN values instead of NaN

## Root Cause

The bug is in how `_valid_sp_values` is used. This property appears to filter out NaN values, treating them as "invalid":

```python
def sum(self, ...):
    valid_vals = self._valid_sp_values
    sp_sum = valid_vals.sum()
    # ...
```

When the array `[nan]` has `fill_value=0.0`, `_valid_sp_values` returns an empty array, so `sum()` returns 0.0 instead of nan.

## Fix

The fix should ensure that NaN values in `sp_values` are included in aggregation calculations. The `_valid_sp_values` property should not filter out NaN when computing aggregations, or the sum/mean methods should use `sp_values` directly and handle NaN propagation correctly.

A potential approach:
1. Change `sum()` and `mean()` to use `self.sp_values` instead of `self._valid_sp_values`
2. Ensure NaN values are properly propagated through the arithmetic operations
3. Maintain the existing `skipna` parameter behavior for intentional NaN exclusion