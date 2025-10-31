# Bug Report: pandas.core.array_algos.masked_reductions sum/prod All Masked Values

**Target**: `pandas.core.array_algos.masked_reductions.sum` and `pandas.core.array_algos.masked_reductions.prod`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When all values are masked and `skipna=True`, `masked_reductions.sum()` returns 0 (the sum identity element) and `masked_reductions.prod()` returns 1 (the product identity element) instead of returning NA. This is inconsistent with other reduction functions (`min`, `max`, `mean`) which correctly return NA when no valid values exist.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st
from pandas.core.array_algos import masked_reductions
from pandas._libs import missing as libmissing


@given(st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1, max_size=100))
def test_sum_with_all_masked(values_list):
    values = np.array(values_list, dtype=np.int64)
    mask = np.ones(len(values), dtype=bool)

    result = masked_reductions.sum(values, mask, skipna=True)

    assert result is libmissing.NA, \
        f"sum with all masked should return NA, got {result}"
```

**Failing input**: Any array with all values masked, e.g., `values_list=[0]`

## Reproducing the Bug

```python
import numpy as np
from pandas.core.array_algos import masked_reductions
from pandas._libs import missing as libmissing

values = np.array([1, 2, 3], dtype=np.int64)
mask_all = np.ones(len(values), dtype=bool)

sum_result = masked_reductions.sum(values, mask_all, skipna=True)
print(f"sum result: {sum_result}, is NA: {sum_result is libmissing.NA}")

prod_result = masked_reductions.prod(values, mask_all, skipna=True)
print(f"prod result: {prod_result}, is NA: {prod_result is libmissing.NA}")

min_result = masked_reductions.min(values, mask_all, skipna=True)
print(f"min result: {min_result}, is NA: {min_result is libmissing.NA}")

max_result = masked_reductions.max(values, mask_all, skipna=True)
print(f"max result: {max_result}, is NA: {max_result is libmissing.NA}")

mean_result = masked_reductions.mean(values, mask_all, skipna=True)
print(f"mean result: {mean_result}, is NA: {mean_result is libmissing.NA}")
```

Output:
```
sum result: 0, is NA: False
prod result: 1, is NA: False
min result: <NA>, is NA: True
max result: <NA>, is NA: True
mean result: <NA>, is NA: True
```

## Why This Is A Bug

1. **Inconsistency**: All other reduction functions (`min`, `max`, `mean`, `var`, `std`) return NA when there are no valid values. `sum` and `prod` should behave consistently.

2. **Semantic incorrectness**: When there are no values to sum, the result should be "not available" (NA), not 0. The value 0 implies "the sum of nothing", which is semantically different from "there are no valid values to sum".

3. **User expectation**: Users expect that when all values are masked/missing, aggregation functions return NA to signal that no valid computation could be performed.

## Root Cause

In `masked_reductions.py`, the `_reductions` function has this logic (lines 54-69):

```python
if not skipna:
    if mask.any() or check_below_min_count(values.shape, None, min_count):
        return libmissing.NA
    else:
        return func(values, axis=axis, **kwargs)
else:
    if check_below_min_count(values.shape, mask, min_count) and (
        axis is None or values.ndim == 1
    ):
        return libmissing.NA

    if values.dtype == np.dtype(object):
        values = values[~mask]
        return func(values, axis=axis, **kwargs)
    return func(values, where=~mask, axis=axis, **kwargs)
```

The problem is at line 69: `return func(values, where=~mask, axis=axis, **kwargs)`

When `min_count=0` (the default), `check_below_min_count` returns `False` even when all values are masked. The code then calls `np.sum(values, where=~mask)`. When `mask` is all `True`, `where=~mask` is all `False`, causing `np.sum` to return 0 (its identity element) and `np.prod` to return 1 (its identity element).

The `check_below_min_count` function only checks if `min_count > 0`, so it doesn't catch the all-masked case when `min_count=0`.

## Fix

Add an explicit check for the all-masked case in `_reductions` before calling the function:

```diff
diff --git a/pandas/core/array_algos/masked_reductions.py b/pandas/core/array_algos/masked_reductions.py
index 1234567..abcdefg 100644
--- a/pandas/core/array_algos/masked_reductions.py
+++ b/pandas/core/array_algos/masked_reductions.py
@@ -62,6 +62,9 @@ def _reductions(
         if check_below_min_count(values.shape, mask, min_count) and (
             axis is None or values.ndim == 1
         ):
             return libmissing.NA
+
+        if mask.all():
+            return libmissing.NA

         if values.dtype == np.dtype(object):
             # object dtype does not support `where` without passing an initial
```

Alternatively, modify `check_below_min_count` to return `True` when all values are masked regardless of `min_count`, or add special handling in `sum` and `prod` functions similar to what `mean`, `var`, and `std` already have (lines 163-164, 176-177, 194-195).