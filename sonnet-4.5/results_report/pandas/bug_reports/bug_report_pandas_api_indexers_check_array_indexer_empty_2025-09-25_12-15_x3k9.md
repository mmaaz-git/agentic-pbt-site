# Bug Report: pandas.api.indexers.check_array_indexer rejects empty float arrays

**Target**: `pandas.api.indexers.check_array_indexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`pandas.api.indexers.check_array_indexer` rejects empty numpy arrays with float dtype, even though:
1. Empty Python lists are accepted and converted to integer arrays
2. Empty integer numpy arrays are accepted
3. An empty array has no elements, so its dtype should not matter

This creates an inconsistency where `np.array([])` (which creates a float64 array by default) is rejected, but `[]` is accepted.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
from hypothesis.extra import numpy as npst
import numpy as np
from pandas.api import indexers

@given(
    npst.arrays(dtype=np.int64, shape=(5,)),
    st.lists(st.integers(min_value=0, max_value=4), min_size=0, max_size=10)
)
def test_check_array_indexer_basic(arr, indices):
    indices_arr = np.array(indices)
    result = indexers.check_array_indexer(arr, indices_arr)
    assert len(result) == len(indices)
```

**Failing input**: `arr=array([0, 0, 0, 0, 0])`, `indices=[]`

## Reproducing the Bug

```python
import numpy as np
from pandas.api import indexers

arr = np.array([1, 2, 3, 4, 5])
empty_float_arr = np.array([])

try:
    result = indexers.check_array_indexer(arr, empty_float_arr)
except IndexError as e:
    print(f"Error: {e}")
```

Output:
```
Error: arrays used as indices must be of integer or boolean type
```

However, these work fine:
```python
indexers.check_array_indexer(arr, [])
indexers.check_array_indexer(arr, np.array([], dtype=np.int64))
```

## Why This Is A Bug

1. **Inconsistency**: Empty Python lists work, but empty numpy float arrays don't
2. **Unexpected behavior**: `np.array([])` naturally creates float64, so users will encounter this
3. **Logical error**: An empty array has no elements to check, so its dtype is irrelevant for validation
4. **Existing code suggests intent**: The function already has special handling for empty arrays (line 531 in source), but only for Python lists, not numpy arrays

## Fix

The fix is to check for empty arrays before checking dtype:

```diff
--- a/pandas/core/indexers/utils.py
+++ b/pandas/core/indexers/utils.py
@@ -528,9 +528,13 @@ def check_array_indexer(array: AnyArrayLike, indexer: Any) -> Any:
     # convert list-likes to array
     if not is_array_like(indexer):
         indexer = pd_array(indexer)
-        if len(indexer) == 0:
-            # empty list is converted to float array by pd.array
-            indexer = np.array([], dtype=np.intp)
+
+    # Handle empty arrays regardless of how they were created
+    # Empty arrays can have any dtype but should be treated as integer arrays for indexing
+    if is_array_like(indexer) and len(indexer) == 0:
+        indexer = np.array([], dtype=np.intp)
+
+    # Now check dtype for non-empty arrays

     dtype = indexer.dtype
     if is_bool_dtype(dtype):
```