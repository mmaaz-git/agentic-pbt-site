# Bug Report: xarray.indexes PandasIndex.roll() ZeroDivisionError on Empty Index

**Target**: `xarray.core.indexes.PandasIndex.roll()`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `PandasIndex.roll()` method crashes with a `ZeroDivisionError` when called on an empty index because it performs modulo by the index length without checking if the length is zero.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas as pd
from xarray.core.indexes import PandasIndex

@st.composite
def xarray_pandas_indexes_including_empty(draw):
    size = draw(st.integers(min_value=0, max_value=100))
    if size == 0:
        pd_index = pd.Index([])
    else:
        values = draw(st.lists(st.integers(), min_size=size, max_size=size))
        pd_index = pd.Index(values)
    dim_name = draw(st.text(min_size=1, max_size=10))
    return PandasIndex(pd_index, dim_name)

@settings(max_examples=200)
@given(xarray_pandas_indexes_including_empty(), st.integers(min_value=-100, max_value=100))
def test_pandasindex_roll_no_crash(index, shift):
    dim = index.dim
    rolled = index.roll({dim: shift})
```

**Failing input**: Any empty pandas index with any shift value, e.g., `PandasIndex(pd.Index([]), "x")` with `shift=1`

## Reproducing the Bug

```python
import pandas as pd
from xarray.core.indexes import PandasIndex

empty_idx = pd.Index([])
xr_idx = PandasIndex(empty_idx, "x")

print(f"Index length: {len(xr_idx.index)}")

rolled = xr_idx.roll({"x": 1})
```

**Output**:
```
Index length: 0
Traceback (most recent call last):
  ...
  File "xarray/core/indexes.py", line 914, in roll
    shift = shifts[self.dim] % self.index.shape[0]
ZeroDivisionError: integer division or modulo by zero
```

## Why This Is A Bug

Rolling an index is a valid operation even on empty indexes. The expected behavior is that rolling an empty index should return an empty index (since there are no elements to roll). Instead, the code crashes with an unhelpful error message.

This violates the principle that operations should handle edge cases gracefully. Empty collections are a common edge case that should be explicitly handled.

## Fix

```diff
diff --git a/xarray/core/indexes.py b/xarray/core/indexes.py
index abc123..def456 100644
--- a/xarray/core/indexes.py
+++ b/xarray/core/indexes.py
@@ -911,7 +911,11 @@ class PandasIndex(Index):
         return {self.dim: get_indexer_nd(self.index, other.index, method, tolerance)}

     def roll(self, shifts: Mapping[Any, int]) -> PandasIndex:
-        shift = shifts[self.dim] % self.index.shape[0]
+        if self.index.shape[0] == 0:
+            # Empty index: nothing to roll
+            return self._replace(self.index[:])
+
+        shift = shifts[self.dim] % self.index.shape[0]

         if shift != 0:
             new_pd_idx = self.index[-shift:].append(self.index[:-shift])
```