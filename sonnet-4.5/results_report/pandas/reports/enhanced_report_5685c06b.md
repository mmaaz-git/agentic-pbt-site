# Bug Report: pandas.api.extensions.take() Crashes with SparseArray Input

**Target**: `pandas.api.extensions.take`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `pandas.api.extensions.take()` function crashes with a TypeError when given a SparseArray input with `allow_fill=False`, despite the function's documentation explicitly listing ExtensionArray as a supported input type.

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


if __name__ == "__main__":
    test_take_with_sparse_array()
```

<details>

<summary>
**Failing input**: `arr=[0]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 18, in <module>
    test_take_with_sparse_array()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 7, in test_take_with_sparse_array
    arr=st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=100),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 12, in test_take_with_sparse_array
    result = take(sparse, indices, allow_fill=False)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/algorithms.py", line 1239, in take
    result = arr.take(indices, axis=axis)
TypeError: SparseArray.take() got an unexpected keyword argument 'axis'
Falsifying example: test_take_with_sparse_array(
    arr=[0],  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import pandas as pd
from pandas.api.extensions import take

sparse = pd.arrays.SparseArray([0, 1, 2, 3, 4])
result = take(sparse, [0, 1, 2], allow_fill=False)
print(result)
```

<details>

<summary>
TypeError: SparseArray.take() got an unexpected keyword argument 'axis'
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/50/repo.py", line 5, in <module>
    result = take(sparse, [0, 1, 2], allow_fill=False)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/algorithms.py", line 1239, in take
    result = arr.take(indices, axis=axis)
TypeError: SparseArray.take() got an unexpected keyword argument 'axis'
```
</details>

## Why This Is A Bug

This violates expected behavior because the `pandas.api.extensions.take()` function's documentation explicitly states that ExtensionArray is a valid and supported input type. SparseArray is a built-in pandas ExtensionArray subclass designed for efficient storage of sparse data, yet the function crashes when processing it.

The root cause is a mismatch between the function's implementation and the ExtensionArray interface. When `allow_fill=False`, the function unconditionally calls `arr.take(indices, axis=axis)` at line 1239 of `pandas/core/algorithms.py`. However, the ExtensionArray base class defines its `take` method signature as `take(self, indices, *, allow_fill=False, fill_value=None)` - notably without an `axis` parameter. SparseArray correctly follows this base class signature and doesn't accept an `axis` parameter, causing the TypeError.

ExtensionArrays are inherently 1-dimensional, so they don't need an axis parameter. The function should handle ExtensionArrays specially and not pass the axis parameter when calling their take method.

## Relevant Context

- **Pandas Version**: 2.3.2
- **Python Version**: 3.13
- **ExtensionArray Documentation**: ExtensionArrays are 1-dimensional arrays that extend pandas functionality. The base class `ExtensionArray.take()` method signature does not include an `axis` parameter.
- **SparseArray Usage**: SparseArray is commonly used for memory-efficient storage of data with many repeated values (typically zeros or NaN). It's a core pandas data structure, not an edge case.
- **Workaround**: Users can directly call `sparse_array.take(indices)` without the axis parameter, bypassing the `pandas.api.extensions.take()` function entirely.
- **Function Location**: The bug is in `/pandas/core/algorithms.py`, line 1239
- **Affected pandas versions**: This affects at least pandas 2.3.2 and likely earlier versions as well

## Proposed Fix

```diff
--- a/pandas/core/algorithms.py
+++ b/pandas/core/algorithms.py
@@ -1236,7 +1236,14 @@ def take(
             arr, indices, axis=axis, allow_fill=True, fill_value=fill_value
         )
     else:
-        result = arr.take(indices, axis=axis)
+        # ExtensionArrays don't accept axis parameter since they're 1D
+        if isinstance(arr, ABCExtensionArray):
+            if axis != 0:
+                raise ValueError(f"axis must be 0 for ExtensionArray, got {axis}")
+            result = arr.take(indices)
+        else:
+            # NumPy style for regular arrays
+            result = arr.take(indices, axis=axis)
     return result
```