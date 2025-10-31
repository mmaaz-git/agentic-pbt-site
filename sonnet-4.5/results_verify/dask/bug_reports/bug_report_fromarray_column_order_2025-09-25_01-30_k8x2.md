# Bug Report: FromArray Column Order Data Corruption

**Target**: `dask.dataframe.dask_expr.io.io.FromArray._column_indices`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `FromArray._column_indices` property does not preserve the order of requested columns, causing silent data corruption where column data is assigned to the wrong column names.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
from dask.dataframe.dask_expr.io.io import FromArray

@given(
    num_columns=st.integers(min_value=3, max_value=10)
)
@settings(max_examples=100)
def test_column_order_preservation(num_columns):
    arr = np.arange(num_columns * 2).reshape(2, num_columns)
    original_columns = [f'col_{i}' for i in range(num_columns)]
    requested_columns = [original_columns[-1], original_columns[0]]

    from_array = FromArray(
        frame=arr,
        chunksize=10,
        original_columns=original_columns,
        meta=None,
        columns=requested_columns
    )

    column_indices = from_array._column_indices
    expected_indices = [num_columns - 1, 0]

    assert column_indices == expected_indices
```

**Failing input**: `num_columns=3`

## Reproducing the Bug

```python
import numpy as np
from dask.dataframe.dask_expr.io.io import FromArray

arr = np.array([[10, 20, 30], [40, 50, 60]])
original_columns = ['a', 'b', 'c']
requested_columns = ['c', 'a']

from_array = FromArray(
    frame=arr,
    chunksize=10,
    original_columns=original_columns,
    meta=None,
    columns=requested_columns
)

column_indices = from_array._column_indices
print(f"Requested columns: {requested_columns}")
print(f"Column indices: {column_indices}")
print(f"Expected indices: [2, 0]")

data_slice = arr[:, column_indices]
print(f"\nActual data slice (wrong): {data_slice}")
print(f"Expected data slice: {arr[:, [2, 0]]}")
```

Output:
```
Requested columns: ['c', 'a']
Column indices: [0, 2]
Expected indices: [2, 0]

Actual data slice (wrong): [[10 30]
 [40 60]]
Expected data slice: [[30 10]
 [60 40]]
```

The data for columns 'a' and 'c' is swapped, causing silent data corruption.

## Why This Is A Bug

When users request columns in a specific order (e.g., `['c', 'a']`), they expect:
1. The resulting DataFrame to have columns in that order
2. Each column to contain the correct data

The current implementation of `_column_indices` (lines 668-676 in `io.py`) iterates through `original_columns` and returns indices in that order, not in the order of the requested columns:

```python
@functools.cached_property
def _column_indices(self):
    if self.operand("columns") is None:
        return slice(0, len(self.original_columns))
    return [
        i
        for i, col in enumerate(self.original_columns)
        if col in self.operand("columns")
    ]
```

This causes `arr[:, [0, 2]]` instead of `arr[:, [2, 0]]`, resulting in columns 'a' and 'c' having swapped data when the DataFrame is constructed with `columns=['c', 'a']` (line 714-718).

## Fix

```diff
--- a/dask/dataframe/dask_expr/io/io.py
+++ b/dask/dataframe/dask_expr/io/io.py
@@ -668,10 +668,8 @@ class FromArray(PartitionsFiltered, BlockwiseIO):
     @functools.cached_property
     def _column_indices(self):
         if self.operand("columns") is None:
             return slice(0, len(self.original_columns))
-        return [
-            i
-            for i, col in enumerate(self.original_columns)
-            if col in self.operand("columns")
-        ]
+        return [self.original_columns.index(col)
+                for col in self.operand("columns")]
```