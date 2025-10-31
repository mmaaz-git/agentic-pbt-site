# Bug Report: SparseArray operations create invalid sparse representation

**Target**: `pandas.core.arrays.sparse.array._sparse_array_op`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When performing binary operations between SparseArrays where one array has no gaps (`ngaps == 0`), the result can have an invalid sparse representation with fill values appearing in `sp_values`, violating the core invariant of sparse arrays.

## Property-Based Test

```python
from pandas.arrays import SparseArray
import numpy as np
from hypothesis import given, strategies as st

@given(
    st.lists(st.integers(min_value=1, max_value=10), min_size=3, max_size=10),
    st.lists(st.integers(min_value=0, max_value=10), min_size=3, max_size=10)
)
def test_sparse_operation_invariant(values1, values2):
    assume(len(values1) == len(values2))
    assume(all(v != 0 for v in values1))  # Ensure left has ngaps==0

    left = SparseArray(values1, fill_value=0)
    right = SparseArray(values2, fill_value=0)
    result = left - right

    # Invariant: sp_values should not contain fill_value
    assert not np.any(result.sp_values == result.fill_value), \
        f"sp_values {result.sp_values} contains fill_value {result.fill_value}"
```

**Failing input**: `values1=[1, 2, 3], values2=[1, 0, 1]`

## Reproducing the Bug

```python
from pandas.arrays import SparseArray
import numpy as np

left = SparseArray([1, 2, 3], fill_value=0)
right = SparseArray([1, 0, 1], fill_value=0)

result = left - right

print(f"result dense: {list(result.to_dense())}")
print(f"result.sp_values: {result.sp_values}")
print(f"result.fill_value: {result.fill_value}")
```

**Output:**
```
result dense: [0, 2, 2]
result.sp_values: [0 2 2]
result.fill_value: 0
```

The first sparse value is `0`, which equals the fill value. This violates the fundamental invariant that `sp_values` should only contain values different from `fill_value`.

## Why This Is A Bug

SparseArrays maintain the invariant that `sp_values` contains only the non-fill values. When `sp_values[i] == fill_value`, the sparse representation is inconsistent and can lead to unexpected behavior in other operations that assume this invariant holds.

The issue occurs in `_sparse_array_op` when one array has `ngaps == 0`. The code converts both arrays to dense, performs the operation, but then incorrectly reuses the sparse index without checking if the result values match the fill value.

## Fix

After computing the dense result when `ngaps == 0`, re-sparsify to ensure the invariant holds:

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -197,12 +197,15 @@ def _sparse_array_op(
     if left.sp_index.ngaps == 0 or right.sp_index.ngaps == 0:
         with np.errstate(all="ignore"):
             result = op(left.to_dense(), right.to_dense())
             fill = op(_get_fill(left), _get_fill(right))

-        if left.sp_index.ngaps == 0:
-            index = left.sp_index
-        else:
-            index = right.sp_index
+        # When one array has no gaps, we computed a dense result
+        # Need to re-sparsify to maintain the invariant that
+        # sp_values doesn't contain fill_value
+        fill_value = lib.item_from_zerodim(fill)
+        result_arr = SparseArray(result, fill_value=fill_value, kind=left.kind)
+        return result_arr
+
     elif left.sp_index.equals(right.sp_index):
         with np.errstate(all="ignore"):
```

This ensures that the result is properly sparsified, removing any fill values from `sp_values`.