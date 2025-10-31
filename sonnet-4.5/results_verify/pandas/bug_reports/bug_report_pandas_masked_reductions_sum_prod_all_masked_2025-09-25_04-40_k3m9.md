# Bug Report: pandas.core.array_algos.masked_reductions Inconsistent NA Handling

**Target**: `pandas.core.array_algos.masked_reductions.sum` and `pandas.core.array_algos.masked_reductions.prod`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `sum` and `prod` functions in `pandas.core.array_algos.masked_reductions` return numeric values (0.0 for sum, 1.0 for prod) when all values are masked, while `mean`, `min`, and `max` correctly return `NA` in the same scenario. This is an inconsistency in the API.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings
from hypothesis.extra import numpy as npst
from pandas.core.array_algos import masked_reductions
from pandas._libs import missing as libmissing

@given(
    values=npst.arrays(
        dtype=np.float64,
        shape=st.integers(min_value=1, max_value=100),
        elements=st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
    ),
)
@settings(max_examples=500)
def test_masked_reduction_all_masked(values):
    mask = np.ones(len(values), dtype=bool)

    assert masked_reductions.sum(values, mask, skipna=True) is libmissing.NA
    assert masked_reductions.prod(values, mask, skipna=True) is libmissing.NA
    assert masked_reductions.min(values, mask, skipna=True) is libmissing.NA
    assert masked_reductions.max(values, mask, skipna=True) is libmissing.NA
    assert masked_reductions.mean(values, mask, skipna=True) is libmissing.NA
```

**Failing input**: `values=array([0.])`

## Reproducing the Bug

```python
import numpy as np
from pandas.core.array_algos import masked_reductions
from pandas._libs import missing as libmissing

values = np.array([0.])
mask = np.array([True])

sum_result = masked_reductions.sum(values, mask, skipna=True)
prod_result = masked_reductions.prod(values, mask, skipna=True)
mean_result = masked_reductions.mean(values, mask, skipna=True)
min_result = masked_reductions.min(values, mask, skipna=True)

print(f"sum:  {sum_result} (expected NA) - Bug: {sum_result is not libmissing.NA}")
print(f"prod: {prod_result} (expected NA) - Bug: {prod_result is not libmissing.NA}")
print(f"mean: {mean_result} (expected NA) - OK: {mean_result is libmissing.NA}")
print(f"min:  {min_result} (expected NA) - OK: {min_result is libmissing.NA}")
```

Output:
```
sum:  0.0 (expected NA) - Bug: True
prod: 1.0 (expected NA) - Bug: True
mean: <NA> (expected NA) - OK: True
min:  <NA> (expected NA) - OK: True
```

## Why This Is A Bug

When all values in an array are masked (indicating all values are missing/NA), reduction operations should return `NA` to indicate that the result cannot be computed. This is the behavior of `mean`, `min`, and `max`, but not `sum` and `prod`.

The current behavior of returning 0.0 for sum and 1.0 for prod (the identity elements) is misleading because it suggests that the sum/product has been computed from actual data, when in fact there is no data to aggregate.

This inconsistency violates the principle of least surprise and makes the API harder to use correctly.

## Fix

The bug is in `pandas/core/array_algos/masked_reductions.py`. The `_reductions` function (used by `sum` and `prod`) should check if all values are masked and return `NA` in that case, similar to how `mean` does it.

```diff
--- a/pandas/core/array_algos/masked_reductions.py
+++ b/pandas/core/array_algos/masked_reductions.py
@@ -59,6 +59,9 @@ def _reductions(
     else:
         if check_below_min_count(values.shape, mask, min_count) and (
             axis is None or values.ndim == 1
         ):
             return libmissing.NA

+        if mask.all():
+            return libmissing.NA
+
         if values.dtype == np.dtype(object):
             # object dtype does not support `where` without passing an initial
             values = values[~mask]
```