# Bug Report: FixedWindowIndexer Violates start <= end Invariant

**Target**: `pandas.core.indexers.objects.FixedWindowIndexer.get_window_bounds`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`FixedWindowIndexer.get_window_bounds` produces window bounds where `start[i] > end[i]`, violating the fundamental invariant that window start must be less than or equal to window end. This occurs when `window_size=0` and `closed='neither'`.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

from hypothesis import given, strategies as st, settings
from pandas.core.indexers.objects import FixedWindowIndexer


@given(
    num_values=st.integers(min_value=0, max_value=100),
    window_size=st.integers(min_value=0, max_value=50),
    center=st.booleans(),
    closed=st.sampled_from([None, "left", "right", "both", "neither"]),
    step=st.one_of(st.none(), st.integers(min_value=1, max_value=10))
)
@settings(max_examples=1000)
def test_fixed_window_indexer_invariants(num_values, window_size, center, closed, step):
    indexer = FixedWindowIndexer(window_size=window_size)
    start, end = indexer.get_window_bounds(
        num_values=num_values,
        center=center,
        closed=closed,
        step=step
    )

    assert len(start) == len(end)

    for i in range(len(start)):
        assert start[i] <= end[i]
        assert 0 <= start[i] <= num_values
        assert 0 <= end[i] <= num_values
```

**Failing input**: `num_values=2, window_size=0, center=False, closed='neither', step=None`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

from pandas.core.indexers.objects import FixedWindowIndexer

indexer = FixedWindowIndexer(window_size=0)
start, end = indexer.get_window_bounds(num_values=2, closed='neither')

print(f"Window size: 0")
print(f"Num values: 2")
print(f"Closed: 'neither'")
print(f"Start: {start}")
print(f"End: {end}")

for i in range(len(start)):
    print(f"Window {i}: start={start[i]}, end={end[i]}, valid={start[i] <= end[i]}")
```

Output:
```
Window size: 0
Num values: 2
Closed: 'neither'
Start: [0 1]
End: [0 0]
Window 0: start=0, end=0, valid=True
Window 1: start=1, end=0, valid=False
```

## Why This Is A Bug

Window bounds represent a slice `[start, end)` where `start` is the inclusive beginning and `end` is the exclusive end. For this to be a valid (possibly empty) range, we must have `start <= end`. When `start > end`, the window bounds are nonsensical and cannot represent any valid slice of data.

This invariant is fundamental to the window indexing system - it's explicitly assumed by code that uses these bounds to slice arrays.

## Fix

The bug occurs when `window_size=0` triggers the centered offset calculation (`offset = -1`), and then `closed='neither'` decrements the end array. The combination results in invalid bounds.

The issue is in line 105-106: when `window_size=0`, the condition `center or self.window_size == 0` always sets `offset = (0 - 1) // 2 = -1`, regardless of whether centering was requested.

```diff
--- a/pandas/core/indexers/objects.py
+++ b/pandas/core/indexers/objects.py
@@ -102,7 +102,7 @@ class FixedWindowIndexer(BaseIndexer):
         closed: str | None = None,
         step: int | None = None,
     ) -> tuple[np.ndarray, np.ndarray]:
-        if center or self.window_size == 0:
+        if center:
             offset = (self.window_size - 1) // 2
         else:
             offset = 0
@@ -114,6 +114,10 @@ class FixedWindowIndexer(BaseIndexer):
         if closed in ["left", "neither"]:
             end -= 1

+        # Ensure start <= end invariant holds
+        for i in range(len(start)):
+            if start[i] > end[i]:
+                end[i] = start[i]
+
         end = np.clip(end, 0, num_values)
         start = np.clip(start, 0, num_values)
```