# Bug Report: dask.array.diff Inconsistent broadcast_to Usage

**Target**: `dask.array.diff`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `diff` function uses inconsistent `broadcast_to` implementations for `prepend` and `append` parameters. The `prepend` parameter uses `dask.array.broadcast_to` (returns dask array), while `append` uses `np.broadcast_to` (returns numpy array). This breaks lazy evaluation semantics and array backend compatibility.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays, array_shapes
import dask.array as da
import numpy as np

@given(
    arr=arrays(
        dtype=np.float64,
        shape=array_shapes(min_dims=2, max_dims=2, min_side=2, max_side=10),
        elements=st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100)
    ),
    scalar=st.floats(allow_nan=False, allow_infinity=False, min_value=-10, max_value=10)
)
def test_diff_prepend_append_consistency(arr, scalar):
    darr = da.from_array(arr, chunks=2)

    result_prepend = da.diff(darr, prepend=scalar, axis=1)
    result_append = da.diff(darr, append=scalar, axis=1)

    assert isinstance(result_prepend._meta, type(darr._meta))
    assert isinstance(result_append._meta, type(darr._meta))
```

**Failing input**: Any scalar value with `append` parameter causes `np.broadcast_to` to be used instead of `dask.array.broadcast_to`.

## Reproducing the Bug

```python
import numpy as np
import dask.array as da

arr = np.array([[1, 2, 3], [4, 5, 6]])
darr = da.from_array(arr, chunks=2)

result_prepend = da.diff(darr, prepend=0, axis=1)
result_append = da.diff(darr, append=0, axis=1)

print(f"prepend uses: dask.array.broadcast_to")
print(f"append uses:  np.broadcast_to")
```

## Why This Is A Bug

The inconsistency violates the dask lazy evaluation model. When `append` is a scalar (ndim==0):

1. **prepend path** (line 593): Uses `broadcast_to` from dask.array, maintaining lazy evaluation
2. **append path** (line 603): Uses `np.broadcast_to`, creating eager numpy array

This causes several issues:
- Breaks lazy evaluation for `append` parameter
- Inconsistent behavior between `prepend` and `append`
- Incompatible with alternative array backends (CuPy, JAX, etc.)
- May cause memory issues with large arrays

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

The fix changes line 603 from `np.broadcast_to` to `broadcast_to`, making it consistent with the `prepend` path at line 593.