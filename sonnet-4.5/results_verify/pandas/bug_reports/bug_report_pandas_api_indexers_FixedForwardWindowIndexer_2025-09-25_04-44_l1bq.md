# Bug Report: pandas.api.indexers.FixedForwardWindowIndexer Negative Window Size

**Target**: `pandas.api.indexers.FixedForwardWindowIndexer.get_window_bounds`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`FixedForwardWindowIndexer.get_window_bounds()` accepts negative `window_size` values without validation, producing invalid window bounds where `end < start`, violating the fundamental invariant of window indexing.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st
from pandas.api.indexers import FixedForwardWindowIndexer


@given(
    num_values=st.integers(min_value=1, max_value=100),
    window_size=st.integers(min_value=-100, max_value=100)
)
def test_fixed_forward_window_negative_size(num_values, window_size):
    indexer = FixedForwardWindowIndexer(window_size=window_size)
    start, end = indexer.get_window_bounds(num_values=num_values)

    assert len(start) == len(end)
    assert np.all(end >= start)
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
print(f"end >= start: {end >= start}")

assert not np.all(end >= start)
```

Output:
```
start: [0 1]
end: [0 0]
end >= start: [ True False]
```

## Why This Is A Bug

The window bounds invariant `end >= start` is fundamental for all window operations - a window cannot end before it starts. The current implementation:

1. Adds negative `window_size` to `start` positions: `end = start + (-1) = [-1, 0, 1, ...]`
2. Clips to valid range: `end = np.clip(end, 0, num_values) = [0, 0, 1, ...]`
3. Produces invalid bounds where `end[1] = 0 < start[1] = 1`

This causes rolling window operations to produce incorrect results (all zeros) rather than failing fast with a clear error message.

## Fix

```diff
--- a/pandas/core/indexers/objects.py
+++ b/pandas/core/indexers/objects.py
@@ -265,6 +265,9 @@ class FixedForwardWindowIndexer(BaseIndexer):
     ) -> tuple[np.ndarray, np.ndarray]:
         if center:
             raise ValueError("Forward-looking windows can't have center=True")
+        if self.window_size < 0:
+            raise ValueError(
+                f"window_size must be non-negative, got {self.window_size}"
+            )
         if closed is not None:
             raise ValueError(
                 "Forward-looking windows don't support setting the closed argument"
```