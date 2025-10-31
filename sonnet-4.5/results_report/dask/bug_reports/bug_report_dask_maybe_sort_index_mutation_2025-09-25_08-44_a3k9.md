# Bug Report: dask.dataframe.utils._maybe_sort Permanently Mutates Index Names

**Target**: `dask.dataframe.utils._maybe_sort`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`_maybe_sort` permanently changes DataFrame index names when they overlap with column names, instead of temporarily renaming them for sorting. This violates the principle of least surprise and causes unexpected side effects.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas as pd
from dask.dataframe.utils import _maybe_sort

@given(st.lists(st.integers(), min_size=2, max_size=10))
def test_maybe_sort_preserves_index_names(data):
    df = pd.DataFrame({'A': data}, index=pd.Index(range(len(data)), name='A'))
    original_name = df.index.names[0]

    result = _maybe_sort(df, check_index=True)

    assert result.index.names[0] == original_name, \
        f"Index name changed from {original_name} to {result.index.names[0]}"
```

**Failing input**: Any DataFrame where `index.name` equals a column name, e.g., `df.index.name = 'A'` and `'A' in df.columns`

## Reproducing the Bug

```python
import pandas as pd
from dask.dataframe.utils import _maybe_sort

df = pd.DataFrame(
    {'A': [2, 1], 'B': [4, 3]},
    index=pd.Index([10, 20], name='A')
)

print(f"Before: df.index.names = {df.index.names}")

result = _maybe_sort(df, check_index=True)

print(f"After: result.index.names = {result.index.names}")
```

**Output:**
```
Before: df.index.names = ['A']
After: result.index.names = ['-overlapped-index-name-0']
```

## Why This Is A Bug

The issue occurs in lines 501-505 of `dask/dataframe/utils.py`:

```python
if set(a.index.names) & set(a.columns):
    a.index.names = [
        "-overlapped-index-name-%d" % i for i in range(len(a.index.names))
    ]
a = a.sort_values(by=methods.tolist(a.columns))
```

The function renames the index to avoid conflicts when sorting, but:
1. It modifies `a.index.names` in place
2. It returns the modified DataFrame without restoring the original names
3. This causes a permanent side effect visible to the caller

The intended behavior appears to be:
- Temporarily rename overlapping index names to allow sorting by columns
- Restore the original index names after sorting

But the actual behavior is:
- Permanently rename the index names
- Return the modified DataFrame

## Fix

Create a copy of the index names and restore them after sorting:

```diff
--- a/dask/dataframe/utils.py
+++ b/dask/dataframe/utils.py
@@ -498,10 +498,14 @@ def _maybe_sort(a, check_index: bool):
     # sort by value, then index
     try:
         if is_dataframe_like(a):
+            original_index_names = None
             if set(a.index.names) & set(a.columns):
+                original_index_names = a.index.names
                 a.index.names = [
                     "-overlapped-index-name-%d" % i for i in range(len(a.index.names))
                 ]
             a = a.sort_values(by=methods.tolist(a.columns))
+            if original_index_names is not None:
+                a.index.names = original_index_names
         else:
             a = a.sort_values()
     except (TypeError, IndexError, ValueError):
```

This ensures that index names are restored after sorting, preventing the side effect while still allowing the sorting operation to succeed.