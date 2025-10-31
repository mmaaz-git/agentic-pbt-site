# Bug Report: FixedForwardWindowIndexer Negative Window Size

**Target**: `pandas.api.indexers.FixedForwardWindowIndexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`FixedForwardWindowIndexer` accepts negative `window_size` values but produces invalid window bounds where `start[i] > end[i]`, violating a fundamental invariant and causing empty windows in rolling operations.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, settings, strategies as st
from pandas.api.indexers import FixedForwardWindowIndexer

@given(
    num_values=st.integers(min_value=1, max_value=100),
    window_size=st.integers(min_value=-100, max_value=-1),
)
@settings(max_examples=500)
def test_fixed_forward_negative_window_size(num_values, window_size):
    indexer = FixedForwardWindowIndexer(window_size=window_size)
    start, end = indexer.get_window_bounds(num_values=num_values)

    for i in range(len(start)):
        assert start[i] <= end[i], f"start[{i}]={start[i]} > end[{i}]={end[i]}"
```

**Failing input**: `num_values=2, window_size=-1`

## Reproducing the Bug

```python
import pandas as pd
from pandas.api.indexers import FixedForwardWindowIndexer

indexer = FixedForwardWindowIndexer(window_size=-1)
start, end = indexer.get_window_bounds(num_values=3)

assert start[1] > end[1]
```

## Why This Is A Bug

1. The invariant `start[i] <= end[i]` must hold for all valid window bounds, but is violated when `window_size < 0`
2. When `start[i] > end[i]`, slicing with `data[start[i]:end[i]]` produces empty windows
3. This causes rolling operations to produce all NaN/zero values instead of raising a clear error
4. `pd.DataFrame.rolling(window=n)` validates that `n >= 0`, but `FixedForwardWindowIndexer` doesn't
5. Negative window size has no semantic meaning for a "forward-looking window"

## Fix

Add validation in `FixedForwardWindowIndexer.__init__` or `get_window_bounds`:

```diff
--- a/pandas/core/indexers/objects.py
+++ b/pandas/core/indexers/objects.py
@@ -100,6 +100,8 @@ class FixedForwardWindowIndexer(BaseIndexer):
         closed: str | None = None,
         step: int | None = None,
     ) -> tuple[np.ndarray, np.ndarray]:
+        if self.window_size < 0:
+            raise ValueError("window_size must be non-negative")
         if center:
             raise ValueError("Forward-looking windows can't have center=True")
         if closed is not None:
```