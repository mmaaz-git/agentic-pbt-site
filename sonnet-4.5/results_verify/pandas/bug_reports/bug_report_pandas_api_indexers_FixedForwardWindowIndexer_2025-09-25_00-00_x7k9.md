# Bug Report: pandas.api.indexers.FixedForwardWindowIndexer Negative Window Size

**Target**: `pandas.api.indexers.FixedForwardWindowIndexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`FixedForwardWindowIndexer.get_window_bounds` produces invalid window bounds (where `end[i] < start[i]`) when initialized with a negative `window_size`, violating the fundamental invariant that window bounds should satisfy `start[i] <= end[i]`.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings
from pandas.api.indexers import FixedForwardWindowIndexer


@settings(max_examples=1000)
@given(
    num_values=st.integers(min_value=1, max_value=100),
    window_size=st.integers(min_value=-100, max_value=100),
    step=st.integers(min_value=1, max_value=10)
)
def test_fixed_forward_window_start_le_end_invariant(num_values, window_size, step):
    indexer = FixedForwardWindowIndexer(window_size=window_size)
    start, end = indexer.get_window_bounds(num_values=num_values, step=step)

    for i in range(len(start)):
        assert start[i] <= end[i], f"Invalid window bounds at index {i}: start={start[i]}, end={end[i]}"
```

**Failing input**: `num_values=2, window_size=-1, step=1`

## Reproducing the Bug

```python
import numpy as np
from pandas.api.indexers import FixedForwardWindowIndexer

indexer = FixedForwardWindowIndexer(window_size=-2)
start, end = indexer.get_window_bounds(num_values=5)

print(f"start: {start}")
print(f"end: {end}")

for i in range(len(start)):
    if start[i] > end[i]:
        print(f"Invalid: start[{i}]={start[i]} > end[{i}]={end[i]}")
```

Output:
```
start: [0 1 2 3 4]
end: [0 0 0 1 2]
Invalid: start[1]=1 > end[1]=0
Invalid: start[2]=2 > end[2]=0
Invalid: start[3]=3 > end[3]=1
Invalid: start[4]=4 > end[4]=2
```

## Why This Is A Bug

Window bounds should always satisfy the invariant `start[i] <= end[i]`. When `window_size` is negative, the implementation computes `end = start + window_size`, which makes `end < start`. The subsequent `np.clip(end, 0, num_values)` doesn't fix this fundamental issue.

This causes downstream failures when used with `df.rolling()`, producing all NaN values instead of raising a clear error.

## Fix

Add validation to reject negative window sizes:

```diff
--- a/pandas/core/indexers/objects.py
+++ b/pandas/core/indexers/objects.py
@@ -264,6 +264,8 @@ class FixedForwardWindowIndexer(BaseIndexer):
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

Alternatively, validation could be added to `BaseIndexer.__init__`:

```diff
--- a/pandas/core/indexers/objects.py
+++ b/pandas/core/indexers/objects.py
@@ -47,6 +47,8 @@ class BaseIndexer:
     def __init__(
         self, index_array: np.ndarray | None = None, window_size: int = 0, **kwargs
     ) -> None:
+        if window_size < 0:
+            raise ValueError("window_size must be non-negative")
         self.index_array = index_array
         self.window_size = window_size
```