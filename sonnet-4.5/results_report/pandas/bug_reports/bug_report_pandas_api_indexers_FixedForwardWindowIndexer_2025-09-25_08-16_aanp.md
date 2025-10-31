# Bug Report: FixedForwardWindowIndexer Negative Window Size

**Target**: `pandas.api.indexers.FixedForwardWindowIndexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`FixedForwardWindowIndexer.get_window_bounds()` produces invalid window bounds when `window_size` is negative, resulting in `start[i] > end[i]` for some indices, which violates the fundamental invariant that window start positions must not exceed window end positions.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st
from pandas.api.indexers import FixedForwardWindowIndexer


@given(st.integers(min_value=1, max_value=100), st.integers(min_value=-100, max_value=-1))
def test_negative_window_size(num_values, window_size):
    indexer = FixedForwardWindowIndexer(window_size=window_size)
    start, end = indexer.get_window_bounds(num_values=num_values)
    assert np.all(start <= end), f"Found start > end"
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
print(f"start[1] = {start[1]}, end[1] = {end[1]}")
assert start[1] <= end[1], "Invariant violated: start[1] > end[1]"
```

Output:
```
start: [0 1]
end: [0 0]
start[1] = 1, end[1] = 0
AssertionError: Invariant violated: start[1] > end[1]
```

## Why This Is A Bug

Window bounds must satisfy `start[i] <= end[i]` for all `i` to represent valid array slices. When `window_size` is negative, the current implementation adds a negative value to the start positions, resulting in end positions that are less than their corresponding start positions. This produces invalid window bounds that cannot meaningfully represent array slices.

The function should either:
1. Validate that `window_size >= 0` and raise an appropriate error for negative values, or
2. Handle negative window sizes correctly (though semantically, negative window sizes don't make sense for forward-looking windows)

## Fix

```diff
--- a/pandas/core/indexers/objects.py
+++ b/pandas/core/indexers/objects.py
@@ -26,6 +26,8 @@ class FixedForwardWindowIndexer(BaseIndexer):
         if closed is not None:
             raise ValueError(
                 "Forward-looking windows don't support setting the closed argument"
             )
+        if self.window_size < 0:
+            raise ValueError("Forward-looking windows require non-negative window_size")
         if step is None:
             step = 1
```