# Bug Report: dask.array.argtopk - Crash When k >= Array Size

**Target**: `dask.array.argtopk`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`da.argtopk` crashes with `ValueError: too many values to unpack` when `k` (the number of elements to extract) equals or exceeds the array size and the array has multiple chunks.

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

**Failing input**: `arr=da.from_array([1,2,3,4,5], chunks=2), k=5`

## Reproducing the Bug

```python
import numpy as np
import dask.array as da

arr = da.from_array(np.array([1, 2, 3, 4, 5], dtype=np.int32), chunks=2)
result = da.argtopk(arr, 5).compute()
```

Output:
```
ValueError: too many values to unpack (expected 2)
```

The error occurs in `dask/array/chunk.py:245` in the `argtopk_aggregate` function when it tries to unpack the result of calling `argtopk`.

## Why This Is A Bug

1. **Valid usage**: Requesting `k` elements when `k == array_size` is a valid operation (return all elements sorted by value)
2. **Inconsistent behavior**: Works with single chunk but fails with multiple chunks
3. **Silent assumption**: The code assumes `k < array_size` but doesn't validate or document this
4. **NumPy compatibility**: `np.argsort` (which `argtopk` aims to optimize) works fine when requesting all elements

From the source code (`dask/array/chunk.py`):

```python
def argtopk(a_plus_idx, k, axis, keepdims):
    ...
    if abs(k) >= a.shape[axis]:
        return a_plus_idx  # BUG: Returns raw input instead of tuple
    ...
    return np.take_along_axis(a, idx2, axis), np.take_along_axis(idx, idx2, axis)
```

The bug: when `k >= a.shape[axis]`, the function returns `a_plus_idx` which can be a list when aggregating multiple chunks. But the caller expects a tuple `(a, idx)`.

## Fix

```diff
def argtopk(a_plus_idx, k, axis, keepdims):
    """Chunk and combine function of argtopk

    Extract the indices of the k largest elements from a on the given axis.
    If k is negative, extract the indices of the -k smallest elements instead.
    Note that, unlike in the parent function, the returned elements
    are not sorted internally.
    """
    assert keepdims is True
    axis = axis[0]

    if isinstance(a_plus_idx, list):
        a_plus_idx = list(flatten(a_plus_idx))
        a = np.concatenate([ai for ai, _ in a_plus_idx], axis)
        idx = np.concatenate(
            [np.broadcast_to(idxi, ai.shape) for ai, idxi in a_plus_idx], axis
        )
    else:
        a, idx = a_plus_idx

    if abs(k) >= a.shape[axis]:
-        return a_plus_idx
+        return a, idx

    idx2 = np.argpartition(a, -k, axis=axis)
    k_slice = slice(-k, None) if k > 0 else slice(-k)
    idx2 = idx2[tuple(k_slice if i == axis else slice(None) for i in range(a.ndim))]
    return np.take_along_axis(a, idx2, axis), np.take_along_axis(idx, idx2, axis)
```