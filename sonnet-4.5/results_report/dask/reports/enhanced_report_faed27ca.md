# Bug Report: dask.array.argtopk - Crash When k >= Array Size with Multiple Chunks

**Target**: `dask.array.argtopk`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`dask.array.argtopk` crashes with `ValueError: too many values to unpack (expected 2)` when k equals or exceeds the array size and the array is split into multiple chunks.

## Property-Based Test

```python
import numpy as np
import dask.array as da
from hypothesis import given, strategies as st, settings
from hypothesis.extra import numpy as hnp


@st.composite
def dask_array_for_argtopk(draw):
    shape = draw(st.tuples(st.integers(min_value=5, max_value=30)))
    dtype = draw(st.sampled_from([np.int32, np.float64]))

    if dtype == np.float64:
        np_arr = draw(hnp.arrays(
            dtype, shape,
            elements=st.floats(min_value=-1000, max_value=1000,
                             allow_nan=False, allow_infinity=False)
        ))
    else:
        np_arr = draw(hnp.arrays(
            dtype, shape,
            elements=st.integers(min_value=-1000, max_value=1000)
        ))

    chunks = draw(st.integers(min_value=2, max_value=max(3, shape[0] // 2)))
    k = draw(st.integers(min_value=1, max_value=min(10, shape[0])))

    return da.from_array(np_arr, chunks=chunks), k


@given(dask_array_for_argtopk())
@settings(max_examples=200)
def test_argtopk_returns_correct_size(data):
    arr, k = data
    result = da.argtopk(arr, k).compute()
    assert len(result) == k
```

<details>

<summary>
**Failing input**: `data=(dask.array<array, shape=(5,), dtype=int32, chunksize=(2,), chunktype=numpy.ndarray>, 5)`
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/58/hypo.py", line 39, in <module>
  |     test_argtopk_returns_correct_size()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/58/hypo.py", line 31, in test_argtopk_returns_correct_size
  |     @settings(max_examples=200)
  |                    ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/58/hypo.py", line 34, in test_argtopk_returns_correct_size
    |     result = da.argtopk(arr, k).compute()
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/dask/base.py", line 373, in compute
    |     (result,) = compute(self, traverse=False, **kwargs)
    |                 ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/dask/base.py", line 681, in compute
    |     results = schedule(expr, keys, **kwargs)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/dask/array/chunk.py", line 249, in argtopk_aggregate
    |     idx = np.take_along_axis(idx, idx2, axis)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/lib/_shape_base_impl.py", line 178, in take_along_axis
    |     axis = normalize_axis_index(axis, arr.ndim)
    |                                       ^^^^^^^^
    | AttributeError: 'tuple' object has no attribute 'ndim'
    | Falsifying example: test_argtopk_returns_correct_size(
    |     data=(dask.array<array, shape=(5,), dtype=int32, chunksize=(3,), chunktype=numpy.ndarray>,
    |      5),
    | )
    | Explanation:
    |     These lines were always and only run by failing examples:
    |         /home/npc/miniconda/lib/python3.13/site-packages/dask/local.py:540
    |         /home/npc/miniconda/lib/python3.13/site-packages/dask/local.py:541
    |         /home/npc/miniconda/lib/python3.13/site-packages/dask/utils.py:2272
    |         /home/npc/miniconda/lib/python3.13/site-packages/dask/utils.py:2277
    |         /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py:54
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/58/hypo.py", line 34, in test_argtopk_returns_correct_size
    |     result = da.argtopk(arr, k).compute()
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/dask/base.py", line 373, in compute
    |     (result,) = compute(self, traverse=False, **kwargs)
    |                 ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/dask/base.py", line 681, in compute
    |     results = schedule(expr, keys, **kwargs)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/dask/array/chunk.py", line 245, in argtopk_aggregate
    |     a, idx = argtopk(a_plus_idx, k, axis, keepdims)
    |     ^^^^^^
    | ValueError: too many values to unpack (expected 2)
    | Falsifying example: test_argtopk_returns_correct_size(
    |     data=(dask.array<array, shape=(5,), dtype=int32, chunksize=(2,), chunktype=numpy.ndarray>,
    |      5),
    | )
    | Explanation:
    |     These lines were always and only run by failing examples:
    |         /home/npc/miniconda/lib/python3.13/site-packages/dask/local.py:262
    |         /home/npc/miniconda/lib/python3.13/site-packages/dask/local.py:540
    |         /home/npc/miniconda/lib/python3.13/site-packages/dask/local.py:541
    |         /home/npc/miniconda/lib/python3.13/site-packages/dask/utils.py:2272
    |         /home/npc/miniconda/lib/python3.13/site-packages/dask/utils.py:2277
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
import numpy as np
import dask.array as da

# Create a simple array
arr = np.array([1, 2, 3, 4, 5], dtype=np.int32)

# Create dask array with multiple chunks
dask_arr = da.from_array(arr, chunks=2)

print(f"Array: {arr}")
print(f"Dask array chunks: {dask_arr.chunks}")
print(f"Requesting k=5 (equal to array size)")

# This should return indices of all 5 elements sorted by their values
# But it crashes when k equals array size with multiple chunks
try:
    result = da.argtopk(dask_arr, 5).compute()
    print(f"Result: {result}")
except Exception as e:
    import traceback
    print(f"\nError occurred: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
```

<details>

<summary>
ValueError: too many values to unpack (expected 2)
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/58/repo.py", line 17, in <module>
    result = da.argtopk(dask_arr, 5).compute()
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/base.py", line 373, in compute
    (result,) = compute(self, traverse=False, **kwargs)
                ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/base.py", line 681, in compute
    results = schedule(expr, keys, **kwargs)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/array/chunk.py", line 245, in argtopk_aggregate
    a, idx = argtopk(a_plus_idx, k, axis, keepdims)
    ^^^^^^
ValueError: too many values to unpack (expected 2)
Array: [1 2 3 4 5]
Dask array chunks: ((2, 2, 1),)
Requesting k=5 (equal to array size)

Error occurred: too many values to unpack (expected 2)

Full traceback:
```
</details>

## Why This Is A Bug

This violates expected behavior in several ways:

1. **Valid Operation**: Requesting k elements when k equals the array size is mathematically valid - it should return all indices sorted by their corresponding values, equivalent to `argsort()`.

2. **Inconsistent Behavior**: The function works correctly when the array has a single chunk but fails with multiple chunks. This inconsistency violates the principle that chunking should be transparent to the user.

3. **Documentation Ambiguity**: The documentation states "Extract the indices of the k largest elements" without specifying that k must be less than the array size. The function docstring mentions "This performs best when k is much smaller than the chunk size" but doesn't forbid k >= array size.

4. **Incorrect Error Handling**: The code at line 228-229 in `/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/array/chunk.py` explicitly checks for `abs(k) >= a.shape[axis]` showing intent to handle this case, but the implementation is incorrect.

5. **Type Inconsistency**: The bug occurs because when k >= array size, the function returns `a_plus_idx` directly, which can be either a tuple `(a, idx)` for single chunks or a list of tuples for multiple chunks. The caller at line 245 always expects to unpack exactly 2 values.

## Relevant Context

The bug is located in the `argtopk` function in `dask/array/chunk.py`. The function serves as both a chunk function and a combine function in the map-reduce pattern used by Dask. When aggregating multiple chunks, `a_plus_idx` becomes a list of tuples, each containing `(array_chunk, indices_chunk)`.

The function correctly handles this by flattening and concatenating when `isinstance(a_plus_idx, list)` is True (lines 219-224), but then incorrectly returns the raw `a_plus_idx` when `abs(k) >= a.shape[axis]` instead of returning the processed `(a, idx)` tuple.

This bug was discovered through property-based testing with Hypothesis, which found two distinct failure modes:
1. `ValueError: too many values to unpack` when returning a list instead of a tuple
2. `AttributeError: 'tuple' object has no attribute 'ndim'` in a related code path

Links:
- Source code: `dask/array/chunk.py:208-234`
- Calling code: `dask/array/chunk.py:237-256` (argtopk_aggregate function)
- Main API: `dask/array/reductions.py:1386-1416` (public argtopk function)

## Proposed Fix

```diff
--- a/dask/array/chunk.py
+++ b/dask/array/chunk.py
@@ -226,7 +226,7 @@ def argtopk(a_plus_idx, k, axis, keepdims):
         a, idx = a_plus_idx

     if abs(k) >= a.shape[axis]:
-        return a_plus_idx
+        return a, idx

     idx2 = np.argpartition(a, -k, axis=axis)
     k_slice = slice(-k, None) if k > 0 else slice(-k)
```