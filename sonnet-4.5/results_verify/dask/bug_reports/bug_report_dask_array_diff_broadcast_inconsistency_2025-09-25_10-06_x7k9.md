# Bug Report: dask.array.diff Inconsistent broadcast_to Usage

**Target**: `dask.array.diff`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `diff` function in `dask.array.routines` uses `broadcast_to` from `dask.array.core` for the `prepend` parameter but uses `np.broadcast_to` from NumPy for the `append` parameter. This inconsistency violates the expected behavior that all intermediate operations should use dask arrays for lazy evaluation.

## Property-Based Test

```python
import numpy as np
import dask.array as da
from hypothesis import given, strategies as st, settings


@given(
    st.lists(st.integers(min_value=-100, max_value=100), min_size=3, max_size=20),
    st.integers(min_value=-50, max_value=50)
)
@settings(max_examples=100)
def test_diff_append_should_use_dask_broadcast(arr_list, append_val):
    arr_np = np.array(arr_list)
    arr_da = da.from_array(arr_np, chunks=len(arr_list) // 2 + 1)

    result = da.diff(arr_da, append=append_val)
    dask_result = result.compute()
    numpy_result = np.diff(arr_np, append=append_val)

    np.testing.assert_array_equal(dask_result, numpy_result)
```

**Failing input**: While the function produces correct results, the implementation violates dask's lazy evaluation contract.

## Reproducing the Bug

```python
import numpy as np
import dask.array as da

arr = da.from_array(np.array([1, 2, 3, 4, 5]), chunks=3)

result_append = da.diff(arr, append=10)

print(type(result_append))
```

The issue is internal: while the function works correctly, the append parameter uses `np.broadcast_to` (line 603 in routines.py) instead of `broadcast_to` from dask.array (as used for prepend on line 593).

## Why This Is A Bug

This violates the consistency principle in the codebase:

1. **Inconsistent with prepend**: Line 593 uses `broadcast_to` from dask.array for prepend, but line 603 uses `np.broadcast_to` for append
2. **Breaks lazy evaluation**: `np.broadcast_to` returns a NumPy array (eager), while `broadcast_to` returns a dask array (lazy)
3. **Code inconsistency**: The same operation should use the same function for both parameters

Evidence from the code (dask/array/routines.py):

Line 593: `prepend = broadcast_to(prepend, tuple(shape))`  (uses dask.array.broadcast_to)
Line 603: `append = np.broadcast_to(append, tuple(shape))`  (uses np.broadcast_to)

The import at line 20 shows `broadcast_to` is imported from `dask.array.core`, which should be used for both cases.

## Fix

```diff
--- a/dask/array/routines.py
+++ b/dask/array/routines.py
@@ -600,7 +600,7 @@ def diff(a, n=1, axis=-1, prepend=None, append=None):
         if append.ndim == 0:
             shape = list(a.shape)
             shape[axis] = 1
-            append = np.broadcast_to(append, tuple(shape))
+            append = broadcast_to(append, tuple(shape))
         combined.append(append)

     if len(combined) > 1:
```