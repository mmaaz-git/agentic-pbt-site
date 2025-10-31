# Bug Report: pandas.core.indexes Variable Scoping Bug

**Target**: `pandas.core.indexes.api.union_indexes._find_common_index_dtype`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The nested function `_find_common_index_dtype` inside `union_indexes` uses the wrong variable name, referencing the outer scope `indexes` instead of its parameter `inds`, leading to incorrect behavior when the function is called with different arguments.

## Property-Based Test

The bug was discovered through code analysis while testing set operation properties on pandas Index objects. The specific property being investigated was dtype preservation during union operations.

## Reproducing the Bug

```python
from pandas import Index

idx1 = Index([1, 2, 3], dtype='int32')
idx2 = Index([4, 5, 6], dtype='int64')

from pandas.core.indexes.api import union_indexes
result = union_indexes([idx1, idx2], sort=False)

print(f"idx1 dtype: {idx1.dtype}")
print(f"idx2 dtype: {idx2.dtype}")
print(f"Result dtype: {result.dtype}")
```

**Location in code**: `/pandas/core/indexes/api.py`, line 265-283

```python
def _find_common_index_dtype(inds):
    """
    Finds a common type for the indexes to pass through to resulting index.

    Parameters
    ----------
    inds: list of Index or list objects

    Returns
    -------
    The common type or None if no indexes were given
    """
    dtypes = [idx.dtype for idx in indexes if isinstance(idx, Index)]  # BUG: uses 'indexes' instead of 'inds'
    if dtypes:
        dtype = find_common_type(dtypes)
    else:
        dtype = None

    return dtype
```

## Why This Is A Bug

1. **Parameter ignored**: The function signature declares parameter `inds` but the implementation uses `indexes` from outer scope
2. **Inconsistent behavior**: The function will use whatever `indexes` is in scope rather than the passed argument
3. **Violates locality**: The nested function doesn't actually use its parameter, making the code misleading
4. **Docstring mismatch**: The docstring says the parameter is `inds: list of Index or list objects` but the code uses `indexes`

While the bug happens to work correctly in the current code (because the function is always called with `indexes` as the argument), it is:
- Confusing for maintainers
- Error-prone if the calling code changes
- Violates the principle of least surprise

## Fix

Change `indexes` to `inds` on line 277:

```diff
 def _find_common_index_dtype(inds):
     """
     Finds a common type for the indexes to pass through to resulting index.

     Parameters
     ----------
     inds: list of Index or list objects

     Returns
     -------
     The common type or None if no indexes were given
     """
-    dtypes = [idx.dtype for idx in indexes if isinstance(idx, Index)]
+    dtypes = [idx.dtype for idx in inds if isinstance(idx, Index)]
     if dtypes:
         dtype = find_common_type(dtypes)
     else:
         dtype = None

     return dtype
```