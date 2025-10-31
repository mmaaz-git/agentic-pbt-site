# Bug Report: xarray.indexes PandasIndex.roll crashes on empty index

**Target**: `xarray.core.indexes.PandasIndex.roll`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `PandasIndex.roll` method crashes with `ZeroDivisionError` when called on an empty index, instead of gracefully handling the empty case.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas as pd
import xarray.indexes as xr_indexes

@given(st.integers(min_value=-100, max_value=100))
def test_pandas_index_roll_on_empty_index(shift):
    """
    Property: roll should work on empty indexes without crashing.
    Rolling an empty index by any amount should return an empty index.
    """
    empty_pd_idx = pd.Index([])
    idx = xr_indexes.PandasIndex(empty_pd_idx, dim='x')

    result = idx.roll({'x': shift})

    assert len(result.index) == 0
    assert result.dim == idx.dim
```

**Failing input**: Any shift value (e.g., `shift=1`)

## Reproducing the Bug

```python
import pandas as pd
import xarray.indexes as xr_indexes

empty_pd_idx = pd.Index([])
idx = xr_indexes.PandasIndex(empty_pd_idx, dim='x')

idx.roll({'x': 1})
```

**Output:**
```
ZeroDivisionError: integer division or modulo by zero
```

**Traceback points to line 914:**
```python
shift = shifts[self.dim] % self.index.shape[0]  # shape[0] is 0!
```

## Why This Is A Bug

1. **Contract violation**: The `roll` method should work on any valid `PandasIndex`, including empty ones. There's no documentation suggesting empty indexes are unsupported.

2. **Real-world impact**: Empty indexes can legitimately occur in data processing pipelines:
   - After filtering operations that remove all elements
   - When working with subsets of data
   - During data cleaning or validation

3. **Expected behavior**: Rolling an empty sequence by any amount should return an empty sequence. This is consistent with how other operations (like slicing) handle empty indexes.

4. **Inconsistent with pandas**: `pandas.Index` itself can be rolled (via slicing operations) even when empty.

## Fix

```diff
--- a/xarray/core/indexes.py
+++ b/xarray/core/indexes.py
@@ -911,7 +911,11 @@ class PandasIndex(Index):
         return {self.dim: get_indexer_nd(self.index, other.index, method, tolerance)}

     def roll(self, shifts: Mapping[Any, int]) -> PandasIndex:
-        shift = shifts[self.dim] % self.index.shape[0]
+        if self.index.shape[0] == 0:
+            # Empty index: rolling has no effect
+            return self._replace(self.index[:])
+
+        shift = shifts[self.dim] % self.index.shape[0]

         if shift != 0:
             new_pd_idx = self.index[-shift:].append(self.index[:-shift])
```

**Alternative fix** (more concise):

```diff
--- a/xarray/core/indexes.py
+++ b/xarray/core/indexes.py
@@ -911,7 +911,10 @@ class PandasIndex(Index):
         return {self.dim: get_indexer_nd(self.index, other.index, method, tolerance)}

     def roll(self, shifts: Mapping[Any, int]) -> PandasIndex:
-        shift = shifts[self.dim] % self.index.shape[0]
+        if self.index.shape[0] == 0:
+            shift = 0
+        else:
+            shift = shifts[self.dim] % self.index.shape[0]

         if shift != 0:
             new_pd_idx = self.index[-shift:].append(self.index[:-shift])
```

Both fixes handle the empty index case by either returning early or setting shift to 0, avoiding the division by zero.