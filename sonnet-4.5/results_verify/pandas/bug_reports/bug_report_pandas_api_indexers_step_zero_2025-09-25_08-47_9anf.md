# Bug Report: FixedForwardWindowIndexer Step Zero Causes Unclear Error

**Target**: `pandas.api.indexers.FixedForwardWindowIndexer.get_window_bounds`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

FixedForwardWindowIndexer.get_window_bounds raises an unclear ZeroDivisionError when step=0 instead of a descriptive ValueError.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pytest
from pandas.api.indexers import FixedForwardWindowIndexer

@given(
    window_size=st.integers(min_value=1, max_value=100),
    num_values=st.integers(min_value=1, max_value=100),
)
def test_fixed_forward_indexer_step_zero_should_raise(window_size, num_values):
    indexer = FixedForwardWindowIndexer(window_size=window_size)

    with pytest.raises(ValueError, match="step must be"):
        indexer.get_window_bounds(num_values=num_values, step=0)
```

**Failing input**: `window_size=1, num_values=1, step=0`

## Reproducing the Bug

```python
from pandas.api.indexers import FixedForwardWindowIndexer

indexer = FixedForwardWindowIndexer(window_size=5)
start, end = indexer.get_window_bounds(num_values=10, step=0)
```

## Why This Is A Bug

1. The function raises `ZeroDivisionError: division by zero` which doesn't clearly explain the problem
2. Users would expect a `ValueError` with a message like "step must be a positive integer"
3. Similar validation exists for other invalid parameters (center=True, closed not None)
4. The error comes from numpy.arange which is an implementation detail users shouldn't see

## Fix

```diff
--- a/pandas/core/indexers/objects.py
+++ b/pandas/core/indexers/objects.py
@@ def get_window_bounds(
         self,
         num_values: int = 0,
         min_periods: int | None = None,
         center: bool | None = None,
         closed: str | None = None,
         step: int | None = None,
     ) -> tuple[np.ndarray, np.ndarray]:
         if center:
             raise ValueError("Forward-looking windows can't have center=True")
         if closed is not None:
             raise ValueError(
                 "Forward-looking windows don't support setting the closed argument"
             )
         if step is None:
             step = 1
+        if step <= 0:
+            raise ValueError("step must be a positive integer")

         start = np.arange(0, num_values, step, dtype="int64")
```