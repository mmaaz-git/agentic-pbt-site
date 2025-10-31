# Bug Report: pandas.api.indexers.check_array_indexer Empty Pandas Array Handling

**Target**: `pandas.api.indexers.check_array_indexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`check_array_indexer` incorrectly rejects empty pandas arrays as invalid indexers, despite correctly handling empty Python lists and having special logic for empty array handling.

## Property-Based Test

```python
from hypothesis import assume, given, settings, strategies as st
import pandas as pd
from pandas.api.indexers import check_array_indexer

@settings(max_examples=1000)
@given(
    array_len=st.integers(min_value=0, max_value=100),
    indexer_data=st.lists(st.booleans(), max_size=100)
)
def test_check_array_indexer_idempotence_boolean(array_len, indexer_data):
    array = pd.array(range(array_len))
    indexer_len = len(indexer_data)
    assume(indexer_len == array_len)

    indexer = pd.array(indexer_data)

    result1 = check_array_indexer(array, indexer)
    result2 = check_array_indexer(array, result1)

    np.testing.assert_array_equal(result1, result2)
```

**Failing input**: `array_len=0, indexer_data=[]`

## Reproducing the Bug

```python
import pandas as pd
from pandas.api.indexers import check_array_indexer

array = pd.array([1, 2, 3])

empty_list = []
result1 = check_array_indexer(array, empty_list)
print(f"Empty list works: {result1}")

empty_pandas_array = pd.array([])
try:
    result2 = check_array_indexer(array, empty_pandas_array)
    print(f"Empty pandas array works: {result2}")
except IndexError as e:
    print(f"Empty pandas array fails: {e}")
```

## Why This Is A Bug

The function has inconsistent behavior for logically equivalent inputs:
- `check_array_indexer(array, [])` succeeds and returns `np.array([], dtype=int64)`
- `check_array_indexer(array, pd.array([]))` fails with `IndexError: arrays used as indices must be of integer or boolean type`

This violates the documented behavior that the function validates array indexers. Empty arrays are valid indexers in numpy/pandas. The code already has special handling for empty arrays (converting them to `dtype=np.intp`), but this logic only applies when `not is_array_like(indexer)`. Since `pd.array([])` is already array-like but has Float64 dtype, it bypasses the empty-array handling and fails the dtype check.

## Fix

```diff
--- a/pandas/core/indexers/utils.py
+++ b/pandas/core/indexers/utils.py
@@ -521,6 +521,10 @@ def check_array_indexer(array: AnyArrayLike, indexer: Any) -> Any:
         if len(indexer) == 0:
             # empty list is converted to float array by pd.array
             indexer = np.array([], dtype=np.intp)
+    elif len(indexer) == 0:
+        # empty array-like indexers should also be converted to integer type
+        # to handle cases where pd.array([]) creates Float64 dtype
+        indexer = np.array([], dtype=np.intp)

     dtype = indexer.dtype
     if is_bool_dtype(dtype):
```