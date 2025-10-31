# Bug Report: FixedForwardWindowIndexer Accepts Invalid Negative Window Sizes

**Target**: `pandas.api.indexers.FixedForwardWindowIndexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`FixedForwardWindowIndexer.get_window_bounds()` accepts negative `window_size` values and produces invalid window bounds where `start[i] > end[i]`, violating the fundamental invariant that window boundaries must satisfy `start <= end`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
from pandas.api.indexers import FixedForwardWindowIndexer


@given(st.integers(min_value=1, max_value=100), st.integers(min_value=-10, max_value=-1))
def test_fixed_forward_window_negative_produces_invalid_bounds(num_values, window_size):
    indexer = FixedForwardWindowIndexer(window_size=window_size)
    start, end = indexer.get_window_bounds(num_values=num_values)

    for i in range(len(start)):
        assert start[i] <= end[i], f"Window bounds violated: start[{i}]={start[i]} > end[{i}]={end[i]}"
```

**Failing input**: `num_values=2, window_size=-1`

## Reproducing the Bug

```python
import numpy as np
from pandas.api.indexers import FixedForwardWindowIndexer

indexer = FixedForwardWindowIndexer(window_size=-1)
start, end = indexer.get_window_bounds(num_values=2)

print(f"start: {start}")
print(f"end: {end}")
print(f"Bug: start[1]={start[1]} > end[1]={end[1]}")
```

Output:
```
start: [0 1]
end: [0 0]
Bug: start[1]=1 > end[1]=0
```

## Why This Is A Bug

The class name `FixedForwardWindowIndexer` implies forward-looking windows, making negative `window_size` semantically invalid. The implementation produces nonsensical window bounds where the start index exceeds the end index, violating the invariant that `start[i] <= end[i]` for all valid window bounds. This would cause incorrect behavior in rolling window operations or crashes when attempting to slice arrays with these bounds.

## Fix

Add validation to reject negative window sizes in `__init__` or `get_window_bounds`:

```diff
--- a/pandas/core/indexers/objects.py
+++ b/pandas/core/indexers/objects.py
@@ -245,6 +245,8 @@ class FixedForwardWindowIndexer(BaseIndexer):
             raise ValueError(
                 "Forward-looking windows don't support setting the closed argument"
             )
+        if self.window_size < 0:
+            raise ValueError("Forward-looking windows require non-negative window_size")
         if step is None:
             step = 1
```