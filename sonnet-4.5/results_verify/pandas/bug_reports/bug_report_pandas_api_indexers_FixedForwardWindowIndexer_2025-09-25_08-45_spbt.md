# Bug Report: FixedForwardWindowIndexer Negative Window Size Invariant Violation

**Target**: `pandas.api.indexers.FixedForwardWindowIndexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

FixedForwardWindowIndexer accepts negative window_size values and produces invalid window bounds where end < start, violating the fundamental window bounds invariant.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
from pandas.api.indexers import FixedForwardWindowIndexer

@given(
    window_size=st.integers(min_value=-100, max_value=100),
    num_values=st.integers(min_value=1, max_value=100),
)
def test_fixed_forward_indexer_start_leq_end_invariant(window_size, num_values):
    indexer = FixedForwardWindowIndexer(window_size=window_size)
    start, end = indexer.get_window_bounds(num_values=num_values)

    assert isinstance(start, np.ndarray)
    assert isinstance(end, np.ndarray)
    assert np.all(start <= end), f"Window bounds invariant violated: start must be <= end for all indices"
```

**Failing input**: `window_size=-1, num_values=2`

## Reproducing the Bug

```python
import numpy as np
from pandas.api.indexers import FixedForwardWindowIndexer

indexer = FixedForwardWindowIndexer(window_size=-1)
start, end = indexer.get_window_bounds(num_values=2)

print(f"start: {start}")
print(f"end: {end}")
print(f"start[1] = {start[1]}, end[1] = {end[1]}")
assert start[1] <= end[1], f"Invariant violated: start[1] ({start[1]}) > end[1] ({end[1]})"
```

## Why This Is A Bug

1. Window bounds should always satisfy `start[i] <= end[i]` - a window cannot have its start index after its end index
2. The standard pandas `rolling(window=n)` validates that `window >= 0` and raises `ValueError: window must be an integer 0 or greater`
3. FixedForwardWindowIndexer should apply the same validation but doesn't
4. The class is documented as creating "fixed-length windows" - negative lengths are nonsensical
5. When used with negative window_size, it silently produces incorrect results instead of raising a clear error

## Fix

Add validation in `FixedForwardWindowIndexer.__init__` or `get_window_bounds` to reject negative window sizes:

```diff
--- a/pandas/core/window/indexers.py
+++ b/pandas/core/window/indexers.py
@@ class FixedForwardWindowIndexer(BaseIndexer):
     def get_window_bounds(
         self,
         num_values: int = 0,
         min_periods: int | None = None,
         center: bool | None = None,
         closed: str | None = None,
         step: int | None = None,
     ) -> tuple[np.ndarray, np.ndarray]:
+        if self.window_size < 0:
+            raise ValueError("window_size must be an integer 0 or greater")
         if center:
             raise ValueError("Forward-looking windows can't have center=True")
         if closed is not None:
```