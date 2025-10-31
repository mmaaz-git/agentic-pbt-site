# Bug Report: FixedForwardWindowIndexer Negative Window Size

**Target**: `pandas.api.indexers.FixedForwardWindowIndexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`FixedForwardWindowIndexer.get_window_bounds()` accepts negative `window_size` values and produces invalid window bounds where `end < start`, violating the fundamental invariant that window boundaries should always satisfy `start <= end`.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st
from pandas.api.indexers import FixedForwardWindowIndexer


@given(
    num_values=st.integers(min_value=1, max_value=100),
    window_size=st.integers(min_value=-100, max_value=100),
    step=st.integers(min_value=1, max_value=10),
)
def test_fixed_forward_start_le_end_always(num_values, window_size, step):
    indexer = FixedForwardWindowIndexer(window_size=window_size)
    start, end = indexer.get_window_bounds(num_values=num_values, step=step)
    assert np.all(start <= end), f"Invariant violated: start > end for window_size={window_size}"
```

**Failing input**: `num_values=2, window_size=-1, step=1`

## Reproducing the Bug

```python
import numpy as np
import pandas as pd
from pandas.api.indexers import FixedForwardWindowIndexer

indexer = FixedForwardWindowIndexer(window_size=-1)
start, end = indexer.get_window_bounds(num_values=2, step=1)

print(f"start: {start}")
print(f"end: {end}")

assert np.all(start <= end)
```

Output:
```
start: [0 1]
end: [0 0]
AssertionError
```

The window bounds are invalid: the second window has `start[1] = 1` but `end[1] = 0`, meaning `start > end`.

When used with actual rolling operations, this produces incorrect results:
```python
df = pd.DataFrame({'values': range(10)})
indexer = FixedForwardWindowIndexer(window_size=-5)
result = df.rolling(indexer).sum()
```

This returns all zeros instead of raising an error or producing meaningful results.

## Why This Is A Bug

1. **Invariant violation**: Window bounds should always satisfy `start[i] <= end[i]` for all windows
2. **No input validation**: The constructor and `get_window_bounds()` don't validate that `window_size >= 0`
3. **Silently produces wrong results**: When used with rolling operations, returns incorrect values (all zeros) instead of raising an error
4. **Inconsistent with documentation**: All examples and docstrings show positive window sizes, suggesting negative values are not intended to be supported

## Fix

Add validation in the constructor or `get_window_bounds()` to ensure `window_size >= 0`:

```diff
--- a/pandas/core/indexers/objects.py
+++ b/pandas/core/indexers/objects.py
@@ -1,5 +1,6 @@
 class FixedForwardWindowIndexer(BaseIndexer):
+    def __init__(self, **kwargs):
+        super().__init__(**kwargs)
+        if self.window_size < 0:
+            raise ValueError(f"window_size must be non-negative, got {self.window_size}")
+
     @Appender(get_window_bounds_doc)
     def get_window_bounds(
         self,
```

Alternatively, validation could be added in `BaseIndexer.__init__()` to prevent negative window sizes across all indexer types.