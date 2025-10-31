# Bug Report: FixedForwardWindowIndexer Negative Window Size Invariant Violation

**Target**: `pandas.api.indexers.FixedForwardWindowIndexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

FixedForwardWindowIndexer with negative window_size produces invalid window bounds where start[i] > end[i], violating the fundamental invariant that window start indices must be less than or equal to end indices.

## Property-Based Test

```python
import numpy as np
import pytest
from hypothesis import given, strategies as st
from pandas.api.indexers import FixedForwardWindowIndexer


@given(st.integers(min_value=1, max_value=100), st.integers(max_value=-1))
def test_fixed_forward_window_negative_size_start_end_invariant(num_values, window_size):
    indexer = FixedForwardWindowIndexer(window_size=window_size)
    start, end = indexer.get_window_bounds(num_values)

    for i in range(len(start)):
        assert start[i] <= end[i], f"Invariant violated: start[{i}]={start[i]} > end[{i}]={end[i]}"
```

**Failing input**: `num_values=2, window_size=-1`

## Reproducing the Bug

```python
from pandas.api.indexers import FixedForwardWindowIndexer

indexer = FixedForwardWindowIndexer(window_size=-1)
start, end = indexer.get_window_bounds(num_values=2)

print(f"start: {start}")
print(f"end: {end}")
print(f"start[1] > end[1]: {start[1]} > {end[1]}")
```

**Output:**
```
start: [0 1]
end: [0 0]
start[1] > end[1]: 1 > 0
```

## Why This Is A Bug

Window bounds should always satisfy the invariant `start[i] <= end[i]` because they represent slice boundaries where `array[start[i]:end[i]]` should be valid. When `start[i] > end[i]`, this produces empty slices in a confusing way and violates user expectations.

The root cause is that negative window_size is not validated or handled correctly. The code computes `end = start + window_size` and then clips to `[0, num_values]`, but clipping negative values to 0 creates inconsistent bounds.

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