# Bug Report: xarray.compat.dask_array_compat.reshape_blockwise (1,1) to (1,) Edge Case

**Target**: `xarray.compat.dask_array_compat.reshape_blockwise`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `reshape_blockwise` function fails to correctly reshape a (1, 1) dask array to (1,), returning (1, 1) instead. This affects the `least_squares` function in `xarray.compat.dask_array_ops` when residuals have shape (1, 1).

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

import dask.array as da
from hypothesis import given, strategies as st, assume
from xarray.compat.dask_array_compat import reshape_blockwise

@given(
    st.integers(min_value=1, max_value=20),
    st.integers(min_value=1, max_value=20)
)
def test_reshape_blockwise_shape_correct(rows, cols):
    total_size = rows * cols
    arr = da.arange(total_size).reshape(rows, cols)
    new_shape = (total_size,)
    reshaped = reshape_blockwise(arr, new_shape)
    assert reshaped.shape == new_shape, f"Shape should match: {reshaped.shape} vs {new_shape}"
```

**Failing input**: `rows=1, cols=1`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

import dask.array as da
from xarray.compat.dask_array_compat import reshape_blockwise

arr = da.arange(1).reshape(1, 1)
result = reshape_blockwise(arr, (1,))

print(f"Expected: (1,)")
print(f"Got: {result.shape}")
assert result.shape == (1,), f"Expected (1,), got {result.shape}"
```

Output:
```
Expected: (1,)
Got: (1, 1)
AssertionError: Expected (1,), got (1, 1)
```

## Why This Is A Bug

The function claims to provide reshape functionality but fails for the specific case of reshaping (1, 1) to (1,). This is a valid reshape operation that should work. The underlying dask `reshape_blockwise` has this limitation, but xarray's wrapper could handle it since the fallback `x.reshape(shape)` works correctly. This edge case can occur in real usage when calling `least_squares` with certain input dimensions where residuals have shape (1, 1).

## Fix

The xarray wrapper should detect this edge case and use the fallback reshape instead of dask's reshape_blockwise:

```diff
--- a/xarray/compat/dask_array_compat.py
+++ b/xarray/compat/dask_array_compat.py
@@ -10,6 +10,11 @@ def reshape_blockwise(
 ):
     if module_available("dask", "2024.08.2"):
         from dask.array import reshape_blockwise
+
+        # Work around dask reshape_blockwise bug with (1, 1) -> (1,)
+        if x.shape == (1, 1) and shape == (1,):
+            return x.reshape(shape)

         return reshape_blockwise(x, shape=shape, chunks=chunks)
     else: