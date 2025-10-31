# Bug Report: xarray.compat.dask_array_compat.reshape_blockwise Fails to Reshape (1,1) to (1,)

**Target**: `xarray.compat.dask_array_compat.reshape_blockwise`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `reshape_blockwise` function incorrectly returns shape (1, 1) instead of (1,) when attempting to reshape a (1, 1) dask array to (1,), violating the expected reshape semantics.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

import dask.array as da
from hypothesis import given, strategies as st
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

# Run the test
if __name__ == "__main__":
    test_reshape_blockwise_shape_correct()
```

<details>

<summary>
**Failing input**: `rows=1, cols=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/32/hypo.py", line 21, in <module>
    test_reshape_blockwise_shape_correct()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/32/hypo.py", line 9, in test_reshape_blockwise_shape_correct
    st.integers(min_value=1, max_value=20),
               ^^^
  File "/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/32/hypo.py", line 17, in test_reshape_blockwise_shape_correct
    assert reshaped.shape == new_shape, f"Shape should match: {reshaped.shape} vs {new_shape}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Shape should match: (1, 1) vs (1,)
Falsifying example: test_reshape_blockwise_shape_correct(
    rows=1,
    cols=1,
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

import dask.array as da
from xarray.compat.dask_array_compat import reshape_blockwise

# Create a (1, 1) dask array
arr = da.arange(1).reshape(1, 1)
print(f"Input array shape: {arr.shape}")
print(f"Input array: {arr.compute()}")

# Try to reshape to (1,)
result = reshape_blockwise(arr, (1,))
print(f"Expected shape: (1,)")
print(f"Actual shape: {result.shape}")

# Verify the failure
try:
    assert result.shape == (1,), f"Expected shape (1,), but got {result.shape}"
    print("Test PASSED")
except AssertionError as e:
    print(f"Test FAILED: {e}")
```

<details>

<summary>
AssertionError: Expected shape (1,), but got (1, 1)
</summary>
```
Input array shape: (1, 1)
Input array: [[0]]
Expected shape: (1,)
Actual shape: (1, 1)
Test FAILED: Expected shape (1,), but got (1, 1)
```
</details>

## Why This Is A Bug

This violates expected reshape behavior in several ways:

1. **Mathematical validity**: Reshaping (1, 1) to (1,) preserves the total number of elements (1 element) and is a standard array operation supported by NumPy and regular dask reshape.

2. **Inconsistent behavior**: The xarray wrapper has a fallback path (`x.reshape(shape)`) that correctly handles this case when dask < 2024.08.2, but fails when using newer dask versions that delegate to `dask.array.reshape_blockwise`.

3. **Function contract violation**: The function name and usage pattern imply standard reshape semantics. There is no documentation indicating that (1, 1) to (1,) reshaping should fail.

4. **Real-world impact**: This bug affects xarray's `least_squares` function in `xarray/compat/dask_array_ops.py` (lines 70-72), where residuals with shape (1, 1) need to be reshaped. The code even contains a comment acknowledging this issue: "Residuals here are (1, 1) but should be (K,) as rhs is (N, K)" (lines 60-61).

## Relevant Context

The bug originates from dask's `reshape_blockwise` implementation but manifests in xarray's compatibility wrapper. The xarray wrapper at `/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages/xarray/compat/dask_array_compat.py` is a thin wrapper that:

1. Checks if dask >= 2024.08.2 is available
2. If yes, delegates to `dask.array.reshape_blockwise` (which has this bug)
3. If no, falls back to `x.reshape(shape)` (which works correctly)

The issue is referenced in dask issue #6516 as noted in the xarray codebase comment. This edge case occurs when dealing with scalar-like results in statistical operations.

## Proposed Fix

The xarray wrapper should detect this specific edge case and use the working fallback reshape:

```diff
--- a/xarray/compat/dask_array_compat.py
+++ b/xarray/compat/dask_array_compat.py
@@ -10,6 +10,10 @@ def reshape_blockwise(
 ):
     if module_available("dask", "2024.08.2"):
         from dask.array import reshape_blockwise
+
+        # Work around dask reshape_blockwise bug with (1, 1) -> (1,)
+        if x.shape == (1, 1) and shape == (1,):
+            return x.reshape(shape)

         return reshape_blockwise(x, shape=shape, chunks=chunks)
     else:
```