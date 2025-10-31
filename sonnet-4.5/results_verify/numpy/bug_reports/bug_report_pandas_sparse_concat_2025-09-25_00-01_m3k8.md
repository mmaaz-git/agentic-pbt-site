# Bug Report: SparseArray concatenation loses data with different fill values

**Target**: `pandas.core.arrays.sparse.SparseArray._concat_same_type`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When concatenating SparseArrays with different `fill_value` settings, data is silently lost. The method only uses the first array's fill value, causing values from other arrays that match their own fill value (but not the first's) to disappear.

## Property-Based Test

```python
from pandas.arrays import SparseArray
import numpy as np
from hypothesis import given, strategies as st, assume

@given(
    st.lists(st.integers(min_value=-10, max_value=10), min_size=2, max_size=10),
    st.lists(st.integers(min_value=-10, max_value=10), min_size=2, max_size=10),
    st.integers(min_value=-10, max_value=10),
    st.integers(min_value=-10, max_value=10)
)
def test_concat_preserves_values(vals1, vals2, fill1, fill2):
    assume(fill1 != fill2)
    arr1 = SparseArray(vals1, fill_value=fill1)
    arr2 = SparseArray(vals2, fill_value=fill2)

    result = SparseArray._concat_same_type([arr1, arr2])
    expected = np.concatenate([arr1.to_dense(), arr2.to_dense()])

    np.testing.assert_array_equal(result.to_dense(), expected)
```

**Failing input**: `vals1=[0, 0, 1], vals2=[2, 2, 3], fill1=0, fill2=2`

## Reproducing the Bug

```python
from pandas.arrays import SparseArray

arr1 = SparseArray([0, 0, 1], fill_value=0)
arr2 = SparseArray([2, 2, 3], fill_value=2)

result = SparseArray._concat_same_type([arr1, arr2])

print(f"arr1: {list(arr1.to_dense())}")
print(f"arr2: {list(arr2.to_dense())}")
print(f"Result: {list(result.to_dense())}")
```

**Expected output:**
```
arr1: [0, 0, 1]
arr2: [2, 2, 3]
Result: [0, 0, 1, 2, 2, 3]
```

**Actual output:**
```
arr1: [0, 0, 1]
arr2: [2, 2, 3]
Result: [0, 0, 1, 0, 0, 3]
```

The values `[2, 2]` from `arr2` are lost because they match `arr2`'s fill value (2), but the concatenation uses `arr1`'s fill value (0), causing these positions to be filled with 0 instead of 2.

## Why This Is A Bug

Concatenation should preserve all values from all input arrays. When arrays have different fill values, the current implementation silently loses data by only using the first array's fill value for the result. This violates the fundamental property: `concat([a, b]).to_dense() == np.concatenate([a.to_dense(), b.to_dense()])`.

## Fix

Before concatenation, convert all arrays to use the same fill value, or convert to dense and reconstruct:

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1181,7 +1181,17 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):
     @classmethod
     def _concat_same_type(cls, to_concat: Sequence[Self]) -> Self:
         fill_value = to_concat[0].fill_value
+
+        # Check if all arrays have the same fill value
+        if not all(arr.fill_value == fill_value for arr in to_concat):
+            # Different fill values - need to handle carefully
+            # Convert all to dense, concatenate, then re-sparsify
+            dense_arrays = [arr.to_dense() for arr in to_concat]
+            concatenated = np.concatenate(dense_arrays)
+            return cls(concatenated, fill_value=fill_value, kind=to_concat[0].kind)

+        # All arrays have the same fill value, can use efficient sparse concat
         values = []
         length = 0
```

This ensures data is never lost during concatenation, at the cost of some performance when fill values differ.