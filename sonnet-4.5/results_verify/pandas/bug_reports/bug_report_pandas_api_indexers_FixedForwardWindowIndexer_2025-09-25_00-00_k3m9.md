# Bug Report: FixedForwardWindowIndexer Invalid Bounds with Negative window_size

**Target**: `pandas.api.indexers.FixedForwardWindowIndexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

FixedForwardWindowIndexer produces invalid window bounds when instantiated with a negative `window_size`, violating the fundamental invariant that `start[i] <= end[i]` for all windows.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.api.indexers import FixedForwardWindowIndexer

@given(
    num_values=st.integers(min_value=1, max_value=100),
    window_size=st.integers(min_value=-20, max_value=-1),
    step=st.integers(min_value=1, max_value=10)
)
def test_fixed_forward_indexer_negative_window_size(num_values, window_size, step):
    indexer = FixedForwardWindowIndexer(window_size=window_size)
    start, end = indexer.get_window_bounds(num_values=num_values, step=step)

    for i in range(len(start)):
        assert start[i] <= end[i], f"Invalid window: start[{i}]={start[i]} > end[{i}]={end[i]}"
```

**Failing input**: `num_values=2, window_size=-1, step=1`

## Reproducing the Bug

```python
from pandas.api.indexers import FixedForwardWindowIndexer

indexer = FixedForwardWindowIndexer(window_size=-1)
start, end = indexer.get_window_bounds(num_values=2, step=1)

print(f"start: {start}")
print(f"end: {end}")
print(f"\nInvariant violated: start[1]={start[1]} > end[1]={end[1]}")
```

Output:
```
start: [0 1]
end: [0 0]

Invariant violated: start[1]=1 > end[1]=0
```

## Why This Is A Bug

The FixedForwardWindowIndexer creates window bounds where `start[i] > end[i]` when given a negative window_size. This violates the fundamental invariant that window bounds should satisfy `start <= end`.

The issue stems from this code in `get_window_bounds`:

```python
start = np.arange(0, num_values, step, dtype="int64")
end = start + self.window_size
if self.window_size:
    end = np.clip(end, 0, num_values)
```

When `window_size` is negative:
1. `end = start + window_size` produces negative values or values less than `start`
2. The `np.clip(end, 0, num_values)` clips negative values to 0
3. This creates situations where `start[i] > end[i]` (e.g., `start[1]=1` but `end[1]=0`)

The class name "FixedForwardWindowIndexer" implies forward-looking windows, which semantically should not support negative window sizes. The code should either:
1. Validate that `window_size >= 0` and raise a descriptive error for negative values
2. Properly handle negative window sizes (though this contradicts the "forward" naming)

## Fix

Add validation in the `__init__` method or `get_window_bounds` to reject negative window sizes:

```diff
--- a/pandas/api/indexers.py
+++ b/pandas/api/indexers.py
@@ -XX,XX +XX,XX @@ class FixedForwardWindowIndexer(BaseIndexer):
     def get_window_bounds(
         self,
         num_values: int = 0,
         min_periods: int | None = None,
         center: bool | None = None,
         closed: str | None = None,
         step: int | None = None,
     ) -> tuple[np.ndarray, np.ndarray]:
+        if self.window_size < 0:
+            raise ValueError("FixedForwardWindowIndexer requires window_size >= 0")
         if center:
             raise ValueError("Forward-looking windows can't have center=True")
```