# Bug Report: pandas.core.indexes.api.union_indexes Fails to Remove Duplicates

**Target**: `pandas.core.indexes.api.union_indexes`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `union_indexes` function fails to remove duplicate values when all input indexes are equal, violating the expected behavior of a union operation which should return unique elements.

## Property-Based Test

```python
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings
from pandas.core.indexes.api import union_indexes
from pandas import Index


@st.composite
def index_strategy(draw):
    values = draw(st.lists(st.integers(min_value=-1000, max_value=1000), min_size=0, max_size=50))
    return Index(values)


@given(index_strategy())
@settings(max_examples=500)
def test_union_indexes_idempotent(idx):
    result = union_indexes([idx, idx])
    assert result.equals(idx.unique())
```

**Failing input**: `Index([0, 0], dtype='int64')`

## Reproducing the Bug

```python
from pandas import Index
from pandas.core.indexes.api import union_indexes

idx_with_dups = Index([0, 0])
result = union_indexes([idx_with_dups, idx_with_dups])
print(f"Result: {list(result)}")
print(f"Expected: [0]")
print(f"Actual contains duplicates: {len(result) > 1}")

idx1 = Index([1, 1])
idx2 = Index([1, 1])
result2 = union_indexes([idx1, idx2])
print(f"union_indexes([Index([1,1]), Index([1,1])]) = {list(result2)}")
print(f"Still has duplicates: {len(result2) > 1}")
```

## Why This Is A Bug

The `union_indexes` function is designed to compute the union of multiple indexes. In set theory and in the pandas API (e.g., `Index.union()`), a union operation should return unique elements. The internal helper function `_unique_indices` (lines 231-263 in api.py) explicitly calls `.unique()` to remove duplicates, indicating this is the intended behavior.

However, the optimization at line 318 of `pandas/core/indexes/api.py` skips calling `_unique_indices` when all input indexes are equal:

```python
if not all(index.equals(other) for other in indexes[1:]):
    index = _unique_indices(indexes, dtype)
```

This means that when all indexes are equal (e.g., `union_indexes([idx, idx])`), the function returns the first index as-is without removing duplicates, even though the union of an index with itself should contain only unique values.

## Fix

```diff
--- a/pandas/core/indexes/api.py
+++ b/pandas/core/indexes/api.py
@@ -315,9 +315,11 @@ def union_indexes(indexes, sort: bool | None = True) -> Index:
     elif kind == "array":
         dtype = _find_common_index_dtype(indexes)
         index = indexes[0]
         if not all(index.equals(other) for other in indexes[1:]):
             index = _unique_indices(indexes, dtype)
+        elif not index.is_unique:
+            # Even if all indexes are equal, we need to remove duplicates
+            index = index.unique()

         name = get_unanimous_names(*indexes)[0]
         if name != index.name:
             index = index.rename(name)
```