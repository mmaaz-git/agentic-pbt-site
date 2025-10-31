# Bug Report: pandas.core.arrays.sparse.SparseArray.astype Returns ndarray

**Target**: `pandas.core.arrays.sparse.SparseArray.astype`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `SparseArray.astype()` method returns a numpy ndarray instead of a SparseArray when the input array contains only fill values, violating its documented contract that "The output will always be a SparseArray."

## Property-Based Test

```python
import numpy as np
from pandas.arrays import SparseArray
from hypothesis import given, strategies as st

@st.composite
def sparse_arrays(draw):
    size = draw(st.integers(min_value=1, max_value=100))
    dtype = draw(st.sampled_from([np.int64, np.float64]))
    data = draw(st.lists(st.integers(min_value=-1000, max_value=1000) if dtype == np.int64
                         else st.floats(allow_nan=False, allow_infinity=False),
                         min_size=size, max_size=size))
    fill_value = 0
    return SparseArray(data, fill_value=fill_value, dtype=dtype)

@given(sparse_arrays())
def test_astype_returns_sparse_array(arr):
    result = arr.astype(np.float64)
    assert isinstance(result, SparseArray), f"Expected SparseArray, got {type(result)}"
```

**Failing input**: `SparseArray([0], fill_value=0, dtype=np.int64)`

## Reproducing the Bug

```python
import numpy as np
from pandas.arrays import SparseArray

arr = SparseArray([0], fill_value=0, dtype=np.int64)
result = arr.astype(np.float64)

print(f"Result type: {type(result)}")
assert isinstance(result, SparseArray), f"Expected SparseArray, got {type(result).__name__}"
```

Output:
```
Result type: <class 'numpy.ndarray'>
AssertionError: Expected SparseArray, got ndarray
```

## Why This Is A Bug

The `astype()` docstring explicitly states: "The output will always be a SparseArray. To convert to a dense ndarray with a certain dtype, use :meth:`numpy.asarray`."

However, when a SparseArray contains only fill values (e.g., `[0]` with `fill_value=0`), calling `astype()` returns a plain numpy ndarray instead of a SparseArray. This violates the documented API contract and breaks code that expects consistent return types.

## Fix

The issue appears to be in the astype implementation around line 620-634 of `/pandas/core/arrays/sparse/array.py`. When the sparse array has no non-fill values, the method should still return a SparseArray, not fall back to a dense array. The fix would ensure that even empty sparse arrays (all fill values) are returned as SparseArray instances.