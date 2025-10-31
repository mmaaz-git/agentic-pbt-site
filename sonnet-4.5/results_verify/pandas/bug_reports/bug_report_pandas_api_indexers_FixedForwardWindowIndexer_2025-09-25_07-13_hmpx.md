# Bug Report: FixedForwardWindowIndexer Invalid Window Bounds with Negative Window Size

**Target**: `pandas.api.indexers.FixedForwardWindowIndexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

FixedForwardWindowIndexer produces invalid window bounds (start > end) when initialized with a negative window_size, violating the fundamental invariant that window start indices must be less than or equal to end indices.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st
from pandas.api.indexers import FixedForwardWindowIndexer

@given(
    num_values=st.integers(min_value=1, max_value=100),
    window_size=st.integers(min_value=-10, max_value=-1),
)
def test_fixed_forward_window_negative_window_size(num_values, window_size):
    indexer = FixedForwardWindowIndexer(window_size=window_size)
    start, end = indexer.get_window_bounds(num_values=num_values)

    for i in range(len(start)):
        assert start[i] <= end[i]
```

**Failing input**: `num_values=2, window_size=-1`

## Reproducing the Bug

```python
from pandas.api.indexers import FixedForwardWindowIndexer

indexer = FixedForwardWindowIndexer(window_size=-1)
start, end = indexer.get_window_bounds(num_values=2)

print(f"start: {start}")
print(f"end: {end}")

assert start[1] > end[1]
```

Output:
```
start: [0 1]
end: [0 0]
```

This also causes incorrect results in actual rolling window operations:

```python
import pandas as pd

df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
indexer = FixedForwardWindowIndexer(window_size=-1)
result = df.rolling(window=indexer).sum()
print(result)
```

Output (all zeros instead of meaningful window sums):
```
     A
0  0.0
1  0.0
2  0.0
3  0.0
4  0.0
```

## Why This Is A Bug

The fundamental invariant for window bounds is that `start[i] <= end[i]` for all indices, as these values are used to slice arrays. When this invariant is violated, the resulting windows are invalid, leading to empty slices and incorrect computation results.

While negative window sizes don't make semantic sense, the API accepts them without validation or error, leading to silent incorrect behavior rather than a clear error message.

## Fix

Add validation in `__init__` or `get_window_bounds` to reject negative window sizes:

```diff
--- a/pandas/core/indexers/objects.py
+++ b/pandas/core/indexers/objects.py
@@ -100,6 +100,8 @@ class FixedForwardWindowIndexer(BaseIndexer):
         if step is None:
             step = 1

+        if self.window_size < 0:
+            raise ValueError(f"window_size must be non-negative, got {self.window_size}")
+
         start = np.arange(0, num_values, step, dtype="int64")
         end = start + self.window_size
         if self.window_size:
```