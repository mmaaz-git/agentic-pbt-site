# Bug Report: pandas.core.util.hashing.combine_hash_arrays Array Length Mismatch

**Target**: `pandas.core.util.hashing.combine_hash_arrays`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`combine_hash_arrays` crashes with a confusing ValueError when given arrays of different lengths, despite not documenting or validating this precondition.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
from pandas.core.util.hashing import hash_array, combine_hash_arrays

@given(
    st.lists(st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1, max_size=100), min_size=1, max_size=10),
)
@settings(max_examples=500)
def test_combine_hash_arrays_assertion(array_lists):
    arrays = [np.array(arr, dtype=np.int64) for arr in array_lists]
    hash_arrays = [hash_array(arr) for arr in arrays]
    num_items = len(hash_arrays)
    result = combine_hash_arrays(iter(hash_arrays), num_items)
    assert result.dtype == np.uint64
```

**Failing input**: `array_lists=[[0], [0, 0]]`

## Reproducing the Bug

```python
import numpy as np
from pandas.core.util.hashing import hash_array, combine_hash_arrays

arr1 = np.array([0], dtype=np.int64)
arr2 = np.array([0, 0], dtype=np.int64)

hash1 = hash_array(arr1)
hash2 = hash_array(arr2)

result = combine_hash_arrays(iter([hash1, hash2]), 2)
```

Running this produces:
```
ValueError: non-broadcastable output operand with shape (1,) doesn't match the broadcast shape (2,)
```

## Why This Is A Bug

The function has an undocumented precondition that all arrays must have the same length. When this precondition is violated, the function crashes at line 74 with a confusing NumPy broadcasting error rather than a clear validation error. The docstring doesn't mention this requirement, making it easy for users to misuse the function.

All internal callers pass same-length arrays (e.g., in `hash_pandas_object`, all column hashes have `len(df.index)` length), but the function itself doesn't validate or document this invariant.

## Fix

Add input validation to provide a clear error message:

```diff
--- a/pandas/core/util/hashing.py
+++ b/pandas/core/util/hashing.py
@@ -47,7 +47,8 @@ def combine_hash_arrays(
     arrays: Iterator[np.ndarray], num_items: int
 ) -> npt.NDArray[np.uint64]:
     """
     Parameters
     ----------
     arrays : Iterator[np.ndarray]
+        All arrays must have the same length.
     num_items : int

     Returns
@@ -69,6 +70,10 @@ def combine_hash_arrays(
     mult = np.uint64(1000003)
     out = np.zeros_like(first) + np.uint64(0x345678)
     last_i = 0
     for i, a in enumerate(arrays):
+        if len(a) != len(out):
+            raise ValueError(
+                f"All arrays must have the same length. Expected {len(out)}, got {len(a)} at position {i}"
+            )
         inverse_i = num_items - i
         out ^= a
         out *= mult
```