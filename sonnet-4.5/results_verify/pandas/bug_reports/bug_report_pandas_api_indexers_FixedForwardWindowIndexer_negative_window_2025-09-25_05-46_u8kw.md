# Bug Report: FixedForwardWindowIndexer Invalid Bounds with Negative window_size

**Target**: `pandas.api.indexers.FixedForwardWindowIndexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`FixedForwardWindowIndexer.get_window_bounds()` produces invalid window bounds (start > end) when initialized with a negative `window_size`, violating the fundamental invariant that window boundaries must satisfy `start[i] <= end[i]`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.api.indexers import FixedForwardWindowIndexer

@given(
    num_values=st.integers(min_value=1, max_value=50),
    window_size=st.integers(min_value=-10, max_value=0)
)
def test_fixed_forward_negative_window_size(num_values, window_size):
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
print(f"\nInvariant violated: start[1] = {start[1]} > end[1] = {end[1]}")
```

Output:
```
start: [0 1]
end: [0 0]

Invariant violated: start[1] = 1 > end[1] = 0
```

## Why This Is A Bug

Window bounds must satisfy the invariant `start[i] <= end[i]` for all valid indices. When `window_size` is negative, the implementation computes `end = start + window_size`, resulting in negative values that get clipped to 0, while `start` continues to increment. This produces invalid bounds where `start[i] > end[i]`.

The class accepts negative `window_size` values without validation in the constructor, but the algorithm in `get_window_bounds()` assumes non-negative values. This leads to logically invalid output that produces incorrect results in rolling window operations (all NaN values).

## Fix

Add validation in the `BaseIndexer.__init__` method to reject negative window sizes:

```diff
--- a/pandas/core/indexers/objects.py
+++ b/pandas/core/indexers/objects.py
@@ -50,6 +50,8 @@ class BaseIndexer:
     def __init__(
         self, index_array: np.ndarray | None = None, window_size: int = 0, **kwargs
     ) -> None:
+        if window_size < 0:
+            raise ValueError("window_size must be non-negative")
         self.index_array = index_array
         self.window_size = window_size
         # Set user defined kwargs as attributes that can be used in get_window_bounds
```