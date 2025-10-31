# Bug Report: pandas.core.array_algos.masked_reductions [sum/prod return identity instead of NA]

**Target**: `pandas.core.array_algos.masked_reductions.sum` and `pandas.core.array_algos.masked_reductions.prod`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When all values in an array are masked (indicating all values should be treated as NA/missing), the `sum()` and `prod()` functions incorrectly return their identity elements (0 for sum, 1 for prod) instead of returning NA. This is inconsistent with `min()` and `max()` which correctly return NA in the same situation.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
import numpy as np
from pandas.core.array_algos.masked_reductions import sum, prod
from pandas._libs import missing as libmissing


@given(
    values=st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=50)
)
@settings(max_examples=500)
def test_masked_reductions_all_masked(values):
    arr = np.array(values)
    mask = np.ones(len(values), dtype=bool)

    result_sum = sum(arr, mask, skipna=True)
    result_prod = prod(arr, mask, skipna=True)

    assert result_sum is libmissing.NA
    assert result_prod is libmissing.NA
```

**Failing input**: `values=[5, 10, 15]` (or any non-empty list)

## Reproducing the Bug

```python
import numpy as np
from pandas.core.array_algos.masked_reductions import sum, prod, min, max
from pandas._libs import missing as libmissing

arr = np.array([5, 10, 15])
mask = np.ones(len(arr), dtype=bool)

result_sum = sum(arr, mask, skipna=True)
result_prod = prod(arr, mask, skipna=True)
result_min = min(arr, mask, skipna=True)
result_max = max(arr, mask, skipna=True)

print(f"sum: {result_sum}")
print(f"prod: {result_prod}")
print(f"min: {result_min}")
print(f"max: {result_max}")
```

**Output**:
```
sum: 0
prod: 1
min: <NA>
max: <NA>
```

## Why This Is A Bug

1. **Inconsistent behavior**: `min()` and `max()` correctly return NA when all values are masked, but `sum()` and `prod()` return identity elements (0 and 1).

2. **Violates expected semantics**: When all values are masked (meaning no valid values exist), reduction operations should return NA, not identity elements. The sum/product of "no values" is undefined, not 0/1.

3. **Breaks mathematical properties**: This violates the property that operations on all-NA data should return NA, which is a fundamental assumption in pandas' missing value handling.

## Fix

The issue is in the `_reductions` function at line 69 in `masked_reductions.py`. When all values are masked, `np.sum(values, where=~mask)` returns 0 (the identity element) because `where` is all False. The function needs to check if all values are masked before calling numpy's reduction function.

```diff
--- a/pandas/core/array_algos/masked_reductions.py
+++ b/pandas/core/array_algos/masked_reductions.py
@@ -59,6 +59,10 @@ def _reductions(
     else:
         if check_below_min_count(values.shape, mask, min_count) and (
             axis is None or values.ndim == 1
+        ):
+            return libmissing.NA
+
+        if mask.all():
             return libmissing.NA

         if values.dtype == np.dtype(object):
```