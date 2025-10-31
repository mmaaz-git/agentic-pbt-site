# Bug Report: pandas.core.array_algos.masked_reductions Inconsistent Behavior for All-Masked Arrays

**Target**: `pandas.core.array_algos.masked_reductions.sum` and `pandas.core.array_algos.masked_reductions.prod`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When all values in an array are masked and `min_count=0` (default), `masked_sum` returns 0 and `masked_prod` returns 1 instead of NA. This is inconsistent with `masked_mean`, `masked_var`, and `masked_std`, which correctly return NA for all-masked arrays.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings
from pandas.core.array_algos.masked_reductions import sum as masked_sum
from pandas._libs import missing as libmissing


@given(
    values=st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1, max_size=100)
)
@settings(max_examples=500)
def test_masked_sum_all_masked_returns_na(values):
    arr = np.array(values, dtype=np.int64)
    mask = np.ones(len(arr), dtype=bool)

    result = masked_sum(arr, mask, skipna=True)
    assert result is libmissing.NA
```

**Failing input**: `values=[0]` (or any other values)

## Reproducing the Bug

```python
import numpy as np
from pandas.core.array_algos.masked_reductions import sum as masked_sum, prod as masked_prod, mean as masked_mean
from pandas._libs import missing as libmissing

arr = np.array([1, 2, 3], dtype=np.int64)
mask = np.ones(3, dtype=bool)

result_sum = masked_sum(arr, mask, skipna=True, min_count=0)
print(f"sum:  {result_sum} (expected NA, got {result_sum})")

result_prod = masked_prod(arr, mask, skipna=True, min_count=0)
print(f"prod: {result_prod} (expected NA, got {result_prod})")

result_mean = masked_mean(arr, mask, skipna=True)
print(f"mean: {result_mean} (expected NA, got {result_mean})")
```

## Why This Is A Bug

When all values are masked (i.e., all missing), the reduction operations should return NA to indicate that no valid computation can be performed. The `mean`, `var`, and `std` functions correctly implement this by checking `mask.all()` before calling the reduction (see line 163 in masked_reductions.py). However, `sum` and `prod` lack this check and instead return the identity elements (0 for sum, 1 for prod) from numpy's `where` parameter behavior. This creates an inconsistency where users get different behaviors for semantically equivalent situations.

## Fix

```diff
--- a/pandas/core/array_algos/masked_reductions.py
+++ b/pandas/core/array_algos/masked_reductions.py
@@ -72,6 +72,8 @@ def sum(
     values: np.ndarray,
     mask: npt.NDArray[np.bool_],
     *,
     skipna: bool = True,
     min_count: int = 0,
     axis: AxisInt | None = None,
 ):
+    if not values.size or mask.all():
+        return libmissing.NA
     return _reductions(
         np.sum, values=values, mask=mask, skipna=skipna, min_count=min_count, axis=axis
     )


 def prod(
     values: np.ndarray,
     mask: npt.NDArray[np.bool_],
     *,
     skipna: bool = True,
     min_count: int = 0,
     axis: AxisInt | None = None,
 ):
+    if not values.size or mask.all():
+        return libmissing.NA
     return _reductions(
         np.prod, values=values, mask=mask, skipna=skipna, min_count=min_count, axis=axis
     )
```