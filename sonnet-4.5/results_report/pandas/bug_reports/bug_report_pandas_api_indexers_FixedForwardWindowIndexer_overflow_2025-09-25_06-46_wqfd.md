# Bug Report: FixedForwardWindowIndexer Integer Overflow

**Target**: `pandas.api.indexers.FixedForwardWindowIndexer`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

FixedForwardWindowIndexer crashes with OverflowError when given extremely large negative window_size values during window bounds calculation.

## Property-Based Test

```python
import pytest
from hypothesis import given, strategies as st
from pandas.api.indexers import FixedForwardWindowIndexer


@given(st.integers(min_value=1, max_value=100), st.integers(max_value=-1))
def test_fixed_forward_window_negative_size_start_end_invariant(num_values, window_size):
    indexer = FixedForwardWindowIndexer(window_size=window_size)
    start, end = indexer.get_window_bounds(num_values)

    for i in range(len(start)):
        assert start[i] <= end[i]
```

**Failing input**: `num_values=1, window_size=-9_223_372_036_854_775_809`

## Reproducing the Bug

```python
from pandas.api.indexers import FixedForwardWindowIndexer

indexer = FixedForwardWindowIndexer(window_size=-9_223_372_036_854_775_809)
start, end = indexer.get_window_bounds(num_values=1)
```

**Output:**
```
OverflowError: Python int too large to convert to C long
```

## Why This Is A Bug

The code attempts to compute `end = start + self.window_size` where start is a numpy int64 array and window_size is a very large negative Python int. This causes an overflow when numpy tries to convert the Python int to a C long for the array operation.

While extremely large negative window_size values are unlikely in real usage, the code should either validate inputs or handle the overflow gracefully rather than crashing.

## Fix

```diff
--- a/pandas/core/indexers/objects.py
+++ b/pandas/core/indexers/objects.py
@@ -330,6 +330,8 @@ class FixedForwardWindowIndexer(BaseIndexer):
         step: int | None = None,
     ) -> tuple[np.ndarray, np.ndarray]:
         if center:
             raise ValueError("Forward-looking windows can't have center=True")
         if closed is not None:
             raise ValueError(
                 "Forward-looking windows don't support setting the closed argument"
             )
+        if self.window_size < 0:
+            raise ValueError("window_size must be non-negative")
         if step is None:
             step = 1
```