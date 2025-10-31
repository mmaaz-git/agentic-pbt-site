# Bug Report: pandas.core.indexers.FixedWindowIndexer Invariant Violation

**Target**: `pandas.core.indexers.objects.FixedWindowIndexer.get_window_bounds`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`FixedWindowIndexer.get_window_bounds` violates the invariant that `start[i] <= end[i]` for all indices when `window_size=0` and `closed='neither'`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pandas.core.indexers.objects import FixedWindowIndexer
import numpy as np

@settings(max_examples=1000)
@given(
    num_values=st.integers(min_value=0, max_value=200),
    window_size=st.integers(min_value=0, max_value=50),
    center=st.booleans(),
    closed=st.sampled_from([None, "left", "right", "both", "neither"]),
    step=st.integers(min_value=1, max_value=10) | st.none(),
)
def test_fixed_window_indexer_comprehensive(num_values, window_size, center, closed, step):
    indexer = FixedWindowIndexer(window_size=window_size)
    start, end = indexer.get_window_bounds(
        num_values=num_values,
        center=center,
        closed=closed,
        step=step
    )

    assert np.all(start <= end), f"start <= end should hold for all windows"
```

**Failing input**: `num_values=2, window_size=0, center=False, closed='neither', step=None`

## Reproducing the Bug

```python
import numpy as np
from pandas.core.indexers.objects import FixedWindowIndexer

indexer = FixedWindowIndexer(window_size=0)
start, end = indexer.get_window_bounds(num_values=2, closed='neither')

print(f"start: {start}")
print(f"end: {end}")
print(f"start[1] > end[1]: {start[1]} > {end[1]}")
```

Output:
```
start: [0 1]
end: [0 0]
start[1] > end[1]: 1 > 0
```

## Why This Is A Bug

Window bounds represent the half-open interval `[start, end)` for slicing operations. The invariant `start <= end` must hold for all window indices, as violating it would mean the window is invalid. When `start > end`, attempting to use these bounds for slicing would result in empty slices at best, or incorrect behavior at worst.

The root cause is in the offset calculation at line 106: when `window_size=0`, `offset = (0-1)//2 = -1`, which causes the `end` array to be decremented below valid values before clipping.

## Fix

```diff
--- a/pandas/core/indexers/objects.py
+++ b/pandas/core/indexers/objects.py
@@ -105,7 +105,10 @@ class FixedWindowIndexer(BaseIndexer):
         if center or self.window_size == 0:
             offset = (self.window_size - 1) // 2
         else:
             offset = 0
+
+        if self.window_size == 0:
+            offset = 0

         end = np.arange(1 + offset, num_values + 1 + offset, step, dtype="int64")
         start = end - self.window_size
```