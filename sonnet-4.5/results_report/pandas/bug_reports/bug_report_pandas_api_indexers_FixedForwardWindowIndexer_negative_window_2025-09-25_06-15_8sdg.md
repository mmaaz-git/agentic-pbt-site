# Bug Report: FixedForwardWindowIndexer Accepts Negative window_size

**Target**: `pandas.api.indexers.FixedForwardWindowIndexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

FixedForwardWindowIndexer accepts negative window_size values and produces invalid window bounds where start[i] > end[i], leading to incorrect rolling window calculations that silently return all NaN values.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
from pandas.api.indexers import FixedForwardWindowIndexer

@given(
    num_values=st.integers(min_value=1, max_value=100),
    window_size=st.integers(min_value=-50, max_value=-1),
    step=st.integers(min_value=1, max_value=10),
)
@settings(max_examples=500)
def test_fixed_forward_window_indexer_negative_window_size(num_values, window_size, step):
    indexer = FixedForwardWindowIndexer(window_size=window_size)
    start, end = indexer.get_window_bounds(num_values=num_values, step=step)

    for i in range(len(start)):
        assert start[i] <= end[i], \
            f"Invalid window: start[{i}]={start[i]} > end[{i}]={end[i]} with window_size={window_size}"
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
print(f"start[1] > end[1]: {start[1]} > {end[1]}")

df = pd.DataFrame({'B': [0, 1, 2, np.nan, 4]})
result = df.rolling(window=indexer, min_periods=1).sum()
print(f"\nRolling result:\n{result}")
```

Output:
```
start: [0 1]
end: [0 0]
start[1] > end[1]: 1 > 0
```

## Why This Is A Bug

FixedForwardWindowIndexer is documented to create "window boundaries for fixed-length windows". A negative window size is semantically invalid - a window cannot have negative length. The implementation fails to validate this precondition, resulting in:

1. Invalid window bounds where start > end
2. Silent failure in rolling operations (returns all NaN)
3. No error message to guide users

Window bounds should always satisfy: 0 <= start[i] <= end[i] <= num_values

## Fix

Add validation in `FixedForwardWindowIndexer.__init__` or at the start of `get_window_bounds`:

```diff
--- a/pandas/core/indexers/objects.py
+++ b/pandas/core/indexers/objects.py
@@ -123,6 +123,8 @@ class FixedForwardWindowIndexer(BaseIndexer):
         closed: str | None = None,
         step: int | None = None,
     ) -> tuple[np.ndarray, np.ndarray]:
+        if self.window_size < 0:
+            raise ValueError("window_size must be non-negative")
         if center:
             raise ValueError("Forward-looking windows can't have center=True")
         if closed is not None:
```