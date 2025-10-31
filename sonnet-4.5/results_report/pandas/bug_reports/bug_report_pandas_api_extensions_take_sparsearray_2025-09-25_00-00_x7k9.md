# Bug Report: pandas.api.extensions.take() Fails with SparseArray

**Target**: `pandas.api.extensions.take`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`pandas.api.extensions.take()` crashes with TypeError when given a SparseArray input and `allow_fill=False`, despite documentation explicitly listing ExtensionArray as a supported input type.

## Property-Based Test

```python
import pandas as pd
from hypothesis import given, strategies as st
from pandas.api.extensions import take


@given(
    arr=st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=100),
)
def test_take_with_sparse_array(arr):
    sparse = pd.arrays.SparseArray(arr)
    indices = [0, 0, 0]
    result = take(sparse, indices, allow_fill=False)
    assert isinstance(result, pd.arrays.SparseArray)
    assert all(r == arr[0] for r in result)
```

**Failing input**: `arr=[0]` (or any list)

## Reproducing the Bug

```python
import pandas as pd
from pandas.api.extensions import take

sparse = pd.arrays.SparseArray([0, 1, 2, 3, 4])
result = take(sparse, [0, 1, 2], allow_fill=False)
```

**Output**:
```
TypeError: SparseArray.take() got an unexpected keyword argument 'axis'
```

## Why This Is A Bug

The `take()` function's docstring explicitly states that ExtensionArray is a valid input type. SparseArray is a built-in pandas ExtensionArray, yet the function crashes when processing it. The issue is on line 1239 of `pandas/core/algorithms.py`:

```python
result = arr.take(indices, axis=axis)
```

The base `ExtensionArray.take()` signature is:
```python
(self, indices, *, allow_fill=False, fill_value=None) -> Self
```

Notice there's no `axis` parameter. SparseArray follows this base signature and doesn't accept `axis`, but `take()` unconditionally passes it when `allow_fill=False`.

## Fix

The base `ExtensionArray` class doesn't include `axis` in its `take()` signature since ExtensionArrays are always 1-D. While some ExtensionArray subclasses add it for compatibility, SparseArray correctly follows the base class signature.

The fix should avoid passing `axis` to ExtensionArrays when `allow_fill=False`:

```diff
--- a/pandas/core/algorithms.py
+++ b/pandas/core/algorithms.py
@@ -1236,7 +1236,14 @@ def take(
             arr, indices, axis=axis, allow_fill=True, fill_value=fill_value
         )
     else:
-        result = arr.take(indices, axis=axis)
+        if isinstance(arr, ABCExtensionArray):
+            if axis != 0:
+                raise ValueError(f"ExtensionArrays are 1-D, axis must be 0, got {axis}")
+            result = arr.take(indices)
+        else:
+            result = arr.take(indices, axis=axis)
     return result
```

Note: ExtensionArray.take() already has `allow_fill=False` as its default, so we don't need to pass it explicitly.