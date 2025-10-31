# Bug Report: dask.bag.Bag.mean Empty Sequence Crash

**Target**: `dask.bag.Bag.mean`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `Bag.mean()` method crashes with `ZeroDivisionError` when called on an empty bag.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import dask.bag as db
import dask

dask.config.set(scheduler='synchronous')

@given(st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100), min_size=0, max_size=100))
def test_mean_no_crash(data):
    b = db.from_sequence(data if data else [0], npartitions=1)
    try:
        mean = b.mean().compute()
    except ZeroDivisionError:
        assert len(data) > 0, "mean() should not crash with ZeroDivisionError on empty sequence"
```

**Failing input**: `[]` (empty list)

## Reproducing the Bug

```python
import dask
import dask.bag as db

dask.config.set(scheduler='synchronous')

b = db.from_sequence([], npartitions=1)
result = b.mean().compute()
```

Output:
```
ZeroDivisionError: float division by zero
```

## Why This Is A Bug

Computing the mean of an empty sequence is mathematically undefined, but the function should handle this gracefully with a clear error message or return a special value (like NaN), rather than crashing with a low-level ZeroDivisionError. This is especially important since `count()` handles empty bags correctly by returning 0.

## Fix

```diff
--- a/dask/bag/core.py
+++ b/dask/bag/core.py
@@ -1132,6 +1132,8 @@ class Bag(DaskMethodsMixin):

     def mean_aggregate(x):
         totals, counts = list(zip(*x))
+        if sum(counts) == 0:
+            raise ValueError("Cannot compute mean of empty sequence")
         return 1.0 * sum(totals) / sum(counts)

     return self.reduction(mean_chunk, mean_aggregate, split_every=False)
```