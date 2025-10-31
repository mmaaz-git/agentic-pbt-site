# Bug Report: SparseArray.astype() Contract Violation

**Target**: `pandas.core.arrays.sparse.SparseArray.astype`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`SparseArray.astype()` violates its documented contract by returning a numpy array instead of a SparseArray when given a non-sparse dtype. The docstring explicitly states "The output will always be a SparseArray", but passing numpy dtypes like `np.float64` or string dtypes like `'float64'` returns a regular numpy.ndarray.

## Property-Based Test

```python
import numpy as np
from pandas.arrays import SparseArray
from hypothesis import given, strategies as st, settings


@given(
    st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=50),
    st.sampled_from([np.int32, np.int64, np.float32, np.float64, 'int32', 'int64', 'float32', 'float64'])
)
@settings(max_examples=100)
def test_astype_always_returns_sparse_array(data, dtype):
    """
    Property: According to the docstring, astype should ALWAYS return a SparseArray.

    From the docstring:
    "The output will always be a SparseArray. To convert to a dense
    ndarray with a certain dtype, use :meth:`numpy.asarray`."
    """
    arr = np.array(data)
    sparse = SparseArray(arr)

    result = sparse.astype(dtype)

    assert isinstance(result, SparseArray), \
        f"astype({dtype}) returned {type(result)}, but docstring says it should ALWAYS return SparseArray"
```

**Failing input**: `data=[0], dtype=np.int32`

## Reproducing the Bug

```python
import numpy as np
from pandas.arrays import SparseArray

sparse = SparseArray([0, 0, 1, 2], fill_value=0)
print(f"Original type: {type(sparse)}")
print(f"Original dtype: {sparse.dtype}")

result = sparse.astype(np.float64)
print(f"Result type: {type(result)}")
print(f"Is SparseArray: {isinstance(result, SparseArray)}")

expected_behavior = "SparseArray"
actual_behavior = type(result).__name__
print(f"\nExpected (per docstring): {expected_behavior}")
print(f"Actual: {actual_behavior}")
```

Output:
```
Original type: <class 'pandas.core.arrays.sparse.array.SparseArray'>
Original dtype: Sparse[int64, 0]
Result type: <class 'numpy.ndarray'>
Is SparseArray: False

Expected (per docstring): SparseArray
Actual: ndarray
```

## Why This Is A Bug

The docstring for `SparseArray.astype()` clearly states:

> "The output will always be a SparseArray. To convert to a dense ndarray with a certain dtype, use :meth:`numpy.asarray`."

However, when passing a non-sparse dtype (like `np.float64`, `'float64'`, etc.), the method returns a numpy.ndarray instead of a SparseArray. This violates the documented contract.

**Current behavior:**
- `astype(SparseDtype(...))` → Returns SparseArray ✓
- `astype('Sparse[float64]')` → Returns SparseArray ✓
- `astype(np.float64)` → Returns numpy.ndarray ✗
- `astype('float64')` → Returns numpy.ndarray ✗

**Expected behavior (per documentation):**
All `astype()` calls should return a SparseArray.

This is a clear API contract violation that can break user code that relies on the documented behavior.

## Fix

The issue appears to be in the implementation of `astype()`. When a non-sparse dtype is provided, the method should wrap the result in a SparseArray instead of returning a raw numpy array.

A potential fix would be to ensure that even when converting to a dense dtype, the result is still wrapped in a SparseArray with appropriate fill_value handling. Alternatively, the docstring should be updated to clarify when numpy arrays are returned (though this would be a breaking change for users relying on the current documentation).

The most user-friendly fix is to update the implementation to match the documentation: always return a SparseArray.