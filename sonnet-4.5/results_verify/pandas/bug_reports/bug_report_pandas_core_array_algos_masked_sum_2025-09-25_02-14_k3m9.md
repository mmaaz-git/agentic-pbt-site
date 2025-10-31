# Bug Report: pandas.core.array_algos.masked_reductions Inconsistent NA Handling

**Target**: `pandas.core.array_algos.masked_reductions.sum` and `pandas.core.array_algos.masked_reductions.prod`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When all values are masked, `sum()` returns 0.0 and `prod()` returns 1.0 instead of NA, creating inconsistency with other reduction functions (`min`, `max`, `mean`) which correctly return NA in the same scenario.

## Property-Based Test

```python
import numpy as np
from hypothesis import assume, given, settings, strategies as st
import pandas.core.array_algos.masked_reductions as masked_reductions
from pandas._libs import missing as libmissing

@given(
    values=st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=1, max_size=100),
    mask_bits=st.lists(st.booleans(), min_size=1, max_size=100)
)
@settings(max_examples=1000)
def test_masked_reductions_consistency(values, mask_bits):
    assume(len(values) == len(mask_bits))
    assume(all(mask_bits))

    arr = np.array(values)
    mask = np.array(mask_bits)

    sum_result = masked_reductions.sum(arr, mask, skipna=True)
    min_result = masked_reductions.min(arr, mask, skipna=True)
    mean_result = masked_reductions.mean(arr, mask, skipna=True)

    assert sum_result is libmissing.NA
    assert min_result is libmissing.NA
    assert mean_result is libmissing.NA
```

**Failing input**: `values=[0.0], mask_bits=[True]`

## Reproducing the Bug

```python
import sys
import numpy as np

sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

import pandas.core.array_algos.masked_reductions as masked_reductions
from pandas._libs import missing as libmissing

arr = np.array([1.0, 2.0, 3.0])
mask_all = np.array([True, True, True])

sum_result = masked_reductions.sum(arr, mask_all, skipna=True, min_count=0)
prod_result = masked_reductions.prod(arr, mask_all, skipna=True, min_count=0)
min_result = masked_reductions.min(arr, mask_all, skipna=True)
mean_result = masked_reductions.mean(arr, mask_all, skipna=True)

print(f"sum:  {sum_result}")
print(f"prod: {prod_result}")
print(f"min:  {min_result}")
print(f"mean: {mean_result}")
```

## Why This Is A Bug

When all values are masked (i.e., no valid values exist), reduction functions should return NA consistently. Currently:

- ✅ `min()` returns NA (line 133: `return libmissing.NA`)
- ✅ `max()` returns NA (line 133: `return libmissing.NA`)
- ✅ `mean()` returns NA (line 164: `if mask.all(): return libmissing.NA`)
- ❌ `sum()` returns 0.0 instead of NA
- ❌ `prod()` returns 1.0 instead of NA

This inconsistency occurs because `sum()` and `prod()` delegate to `_reductions()`, which only checks `min_count` to decide whether to return NA. When `min_count=0` (the default), it calls `np.sum(values, where=~mask)` which returns the identity element (0 for sum, 1 for prod) when summing/multiplying zero values.

## Fix

```diff
--- a/pandas/core/array_algos/masked_reductions.py
+++ b/pandas/core/array_algos/masked_reductions.py
@@ -57,6 +57,10 @@ def _reductions(
         else:
             return func(values, axis=axis, **kwargs)
     else:
+        if mask.all():
+            return libmissing.NA
+
         if check_below_min_count(values.shape, mask, min_count) and (
             axis is None or values.ndim == 1
         ):
```