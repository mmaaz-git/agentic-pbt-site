# Bug Report: dask.array.ravel() crashes on empty arrays with non-zero dimensions

**Target**: `dask.array.ravel()` and `dask.array.reshape()`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`dask.array.ravel()` crashes with `TypeError: reduce() of empty iterable with no initial value` when called on empty arrays that have non-zero dimensions (e.g., shape `(3, 0)`). NumPy correctly handles these arrays, but Dask fails due to missing initial value in `reduce()` calls.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from hypothesis.extra.numpy import arrays, array_shapes
import dask.array as da
import numpy as np


@given(
    arr=arrays(
        dtype=np.float64,
        shape=array_shapes(min_dims=1, max_dims=2, min_side=0, max_side=10)
    )
)
@settings(max_examples=100, deadline=None)
def test_ravel_preserves_elements(arr):
    darr = da.from_array(arr, chunks=2)
    raveled = da.ravel(darr)
    assert raveled.ndim == 1
    assert raveled.size == arr.size
    result = raveled.compute()
    expected = np.ravel(arr)
    np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    # Run the test to find failing cases
    test_ravel_preserves_elements()
```

<details>

<summary>
**Failing input**: `array([], shape=(3, 0), dtype=float64)`
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/15/hypo.py", line 26, in <module>
  |     test_ravel_preserves_elements()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/15/hypo.py", line 8, in test_ravel_preserves_elements
  |     arr=arrays(
  |
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/15/hypo.py", line 16, in test_ravel_preserves_elements
    |     raveled = da.ravel(darr)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/dask/array/routines.py", line 1925, in ravel
    |     return asanyarray(array_like).reshape((-1,))
    |            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/dask/array/core.py", line 2274, in reshape
    |     return reshape(self, shape, merge_chunks=merge_chunks, limit=limit)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/dask/array/reshape.py", line 362, in reshape
    |     inchunks, outchunks, _, _ = reshape_rechunk(x.shape, shape, x.chunks)
    |                                 ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/dask/array/reshape.py", line 105, in reshape_rechunk
    |     if reduce(mul, outshape[oleft : oi + 1]) != din:
    |        ~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    | TypeError: reduce() of empty iterable with no initial value
    | Falsifying example: test_ravel_preserves_elements(
    |     arr=array([], shape=(3, 0), dtype=float64),
    | )
    | Explanation:
    |     These lines were always and only run by failing examples:
    |         /home/npc/miniconda/lib/python3.13/site-packages/dask/array/reshape.py:43
    |         /home/npc/miniconda/lib/python3.13/site-packages/dask/array/reshape.py:97
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/15/hypo.py", line 16, in test_ravel_preserves_elements
    |     raveled = da.ravel(darr)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/dask/array/routines.py", line 1925, in ravel
    |     return asanyarray(array_like).reshape((-1,))
    |            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/dask/array/core.py", line 2274, in reshape
    |     return reshape(self, shape, merge_chunks=merge_chunks, limit=limit)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/dask/array/reshape.py", line 362, in reshape
    |     inchunks, outchunks, _, _ = reshape_rechunk(x.shape, shape, x.chunks)
    |                                 ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/dask/array/reshape.py", line 106, in reshape_rechunk
    |     raise NotImplementedError(_not_implemented_message)
    | NotImplementedError:
    | Dask's reshape only supports operations that merge or split existing dimensions
    | evenly. For example:
    |
    | >>> x = da.ones((6, 5, 4), chunks=(3, 2, 2))
    | >>> x.reshape((3, 2, 5, 4))  # supported, splits 6 into 3 & 2
    | >>> x.reshape((30, 4))       # supported, merges 6 & 5 into 30
    | >>> x.reshape((4, 5, 6))     # unsupported, existing dimensions split unevenly
    |
    | To work around this you may call reshape in multiple passes, or (if your data
    | is small enough) call ``compute`` first and handle reshaping in ``numpy``
    | directly.
    |
    | Falsifying example: test_ravel_preserves_elements(
    |     arr=array([], shape=(0, 3), dtype=float64),
    | )
    | Explanation:
    |     These lines were always and only run by failing examples:
    |         /home/npc/miniconda/lib/python3.13/site-packages/dask/array/reshape.py:97
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
import dask.array as da
import numpy as np
import traceback
import sys

# Create an empty array with non-zero dimensions
arr = np.empty((3, 0), dtype=np.float64)
print(f"Input array shape: {arr.shape}")
print(f"Input array size: {arr.size}")

# Demonstrate NumPy behavior (works correctly)
np_raveled = np.ravel(arr)
print(f"\nNumPy ravel result shape: {np_raveled.shape}")
print(f"NumPy ravel result: {np_raveled}")

# Try with Dask (should crash)
print("\nAttempting Dask ravel...")
darr = da.from_array(arr, chunks=2)
print(f"Dask array created with chunks={darr.chunks}")

try:
    raveled = da.ravel(darr)
    print(f"Dask ravel created (lazy evaluation)")
    result = raveled.compute()
    print(f"Dask ravel result shape: {result.shape}")
    print(f"Dask ravel result: {result}")
except Exception as e:
    print(f"\nDask ravel failed with error:")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")

    # Print full traceback
    print("\nFull traceback:")
    traceback.print_exc(file=sys.stdout)
```

<details>

<summary>
TypeError: reduce() of empty iterable with no initial value
</summary>
```
Input array shape: (3, 0)
Input array size: 0

NumPy ravel result shape: (0,)
NumPy ravel result: []

Attempting Dask ravel...
Dask array created with chunks=((2, 1), (0,))

Dask ravel failed with error:
Error type: TypeError
Error message: reduce() of empty iterable with no initial value

Full traceback:
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/15/repo.py", line 22, in <module>
    raveled = da.ravel(darr)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/array/routines.py", line 1925, in ravel
    return asanyarray(array_like).reshape((-1,))
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/array/core.py", line 2274, in reshape
    return reshape(self, shape, merge_chunks=merge_chunks, limit=limit)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/array/reshape.py", line 362, in reshape
    inchunks, outchunks, _, _ = reshape_rechunk(x.shape, shape, x.chunks)
                                ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/array/reshape.py", line 105, in reshape_rechunk
    if reduce(mul, outshape[oleft : oi + 1]) != din:
       ~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: reduce() of empty iterable with no initial value
```
</details>

## Why This Is A Bug

This violates expected behavior for several specific reasons:

1. **NumPy API Compatibility**: The `dask.array.ravel()` function is decorated with `@derived_from(np)` at line 1923 of `/home/npc/miniconda/lib/python3.13/site-packages/dask/array/routines.py`, explicitly indicating it should match NumPy's behavior. NumPy successfully handles all empty array configurations, always returning shape `(0,)`.

2. **Valid Input Rejection**: Empty arrays with non-zero dimensions like `(3, 0)` or `(0, 3)` are valid NumPy arrays that occur naturally in data processing (e.g., after filtering operations that remove all elements). These are not pathological edge cases.

3. **Inconsistent Behavior**: The function works correctly for fully empty arrays like `(0,)` and `(0, 0)` but fails for partially empty arrays like `(3, 0)`. This inconsistency indicates an implementation bug rather than intentional design.

4. **Python Programming Error**: The crash occurs at line 105 in `reshape_rechunk()` when calling `reduce(mul, outshape[oleft : oi + 1])` without an initial value. This is a common Python mistake - `reduce()` requires an initial value when the iterable might be empty. The multiplicative identity (1) should be used.

5. **Fundamental Operation Failure**: `ravel()` is one of the most basic array operations - it simply flattens an array to 1D. There's no mathematical or computational reason why it shouldn't work on empty arrays.

## Relevant Context

The bug manifests in two different ways depending on the empty array configuration:

1. **Shape `(3, 0)` and similar**: Crashes with `TypeError` at line 105 due to empty slice in `reduce()`
2. **Shape `(0, 3)` and similar**: Raises `NotImplementedError` at line 106 after the reduce check fails

The root cause is in the `reshape_rechunk()` function in `/home/npc/miniconda/lib/python3.13/site-packages/dask/array/reshape.py`. The function uses `reduce(mul, ...)` in multiple places (lines 62, 68, 103, 105) to calculate the product of dimensions, but doesn't provide the required initial value when the sequence might be empty.

The Dask documentation acknowledges incomplete NumPy compatibility but doesn't explicitly exclude empty arrays from supported operations. The `@derived_from(np)` decorator on `ravel()` creates a reasonable expectation that it should handle all valid NumPy arrays.

Documentation link: https://docs.dask.org/en/stable/array.html

## Proposed Fix

```diff
--- a/dask/array/reshape.py
+++ b/dask/array/reshape.py
@@ -59,7 +59,7 @@ def reshape_rechunk(inshape, outshape, inchunks, disallow_dimension_expansion=F
             ileft = ii - 1
             mapper_in[ii] = oi
             while (
-                ileft >= 0 and reduce(mul, inshape[ileft : ii + 1]) < dout
+                ileft >= 0 and reduce(mul, inshape[ileft : ii + 1], 1) < dout
             ):  # 4 < 64, 4*4 < 64, 4*4*4 == 64
                 mapper_in[ileft] = oi
                 ileft -= 1
@@ -65,7 +65,7 @@ def reshape_rechunk(inshape, outshape, inchunks, disallow_dimension_expansion=F
                 ileft -= 1

             mapper_in[ileft] = oi
-            if reduce(mul, inshape[ileft : ii + 1]) != dout:
+            if reduce(mul, inshape[ileft : ii + 1], 1) != dout:
                 raise NotImplementedError(_not_implemented_message)
             # Special case to avoid intermediate rechunking:
             # When all the lower axis are completely chunked (chunksize=1) then
@@ -100,8 +100,8 @@ def reshape_rechunk(inshape, outshape, inchunks, disallow_dimension_expansion=F
                     "reshape_blockwise not implemented for expanding dimensions without passing chunk hints."
                 )
             oleft = oi - 1
-            while oleft >= 0 and reduce(mul, outshape[oleft : oi + 1]) < din:
+            while oleft >= 0 and reduce(mul, outshape[oleft : oi + 1], 1) < din:
                 oleft -= 1
-            if reduce(mul, outshape[oleft : oi + 1]) != din:
+            if reduce(mul, outshape[oleft : oi + 1], 1) != din:
                 raise NotImplementedError(_not_implemented_message)
             # TODO: don't coalesce shapes unnecessarily
```