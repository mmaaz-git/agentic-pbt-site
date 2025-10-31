# Bug Report: pandas.core.arrays.SparseArray Silent Data Corruption When Changing Fill Value

**Target**: `pandas.core.arrays.SparseArray`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When constructing a SparseArray from another SparseArray with a different fill_value, all data values that were equal to the original fill_value are silently corrupted and replaced with the new fill value, resulting in complete data loss.

## Property-Based Test

```python
import numpy as np
from pandas.core.arrays import SparseArray
from hypothesis import given, strategies as st, settings


@given(
    data=st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False), min_size=5, max_size=50),
    old_fill=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
    new_fill=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=500)
def test_sparsearray_change_fill_value(data, old_fill, new_fill):
    sparse = SparseArray(data, fill_value=old_fill)
    original_dense = sparse.to_dense()

    new_sparse = SparseArray(sparse, fill_value=new_fill)

    assert np.allclose(new_sparse.to_dense(), original_dense, equal_nan=True, rtol=1e-10)


if __name__ == "__main__":
    # Run the test
    test_sparsearray_change_fill_value()
```

<details>

<summary>
**Failing input**: `data=[0.0, 0.0, 0.0, 0.0, 0.0]`, `old_fill=0.0`, `new_fill=1.0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 23, in <module>
    test_sparsearray_change_fill_value()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 7, in test_sparsearray_change_fill_value
    data=st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False), min_size=5, max_size=50),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 18, in test_sparsearray_change_fill_value
    assert np.allclose(new_sparse.to_dense(), original_dense, equal_nan=True, rtol=1e-10)
           ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_sparsearray_change_fill_value(
    data=[0.0, 0.0, 0.0, 0.0, 0.0],
    old_fill=0.0,
    new_fill=1.0,
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
from pandas.core.arrays import SparseArray

# Create a SparseArray with values equal to the fill_value
sparse = SparseArray([0.0, 0.0, 0.0, 0.0, 0.0], fill_value=0.0)
print(f"Original sparse array: {sparse.to_dense()}")
print(f"Original fill_value: {sparse.fill_value}")
print(f"Original sparse values: {sparse.sp_values}")
print(f"Original sparse index: {sparse.sp_index}")
print()

# Create a new SparseArray from the first one with different fill_value
new_sparse = SparseArray(sparse, fill_value=1.0)
print(f"After changing fill_value to 1.0: {new_sparse.to_dense()}")
print(f"New fill_value: {new_sparse.fill_value}")
print(f"New sparse values: {new_sparse.sp_values}")
print(f"New sparse index: {new_sparse.sp_index}")
print()

# Check if the values are preserved
expected = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
actual = new_sparse.to_dense()
print(f"Expected: {expected}")
print(f"Actual: {actual}")
print(f"Are they equal? {np.allclose(expected, actual)}")

# This assertion will fail
assert np.allclose(new_sparse.to_dense(), [0.0, 0.0, 0.0, 0.0, 0.0]), \
    f"Expected {[0.0, 0.0, 0.0, 0.0, 0.0]} but got {new_sparse.to_dense()}"
```

<details>

<summary>
AssertionError: Data values changed from [0.0, 0.0, 0.0, 0.0, 0.0] to [1.0, 1.0, 1.0, 1.0, 1.0]
</summary>
```
Original sparse array: [0. 0. 0. 0. 0.]
Original fill_value: 0.0
Original sparse values: []
Original sparse index: IntIndex
Indices: array([], dtype=int32)


After changing fill_value to 1.0: [1. 1. 1. 1. 1.]
New fill_value: 1.0
New sparse values: []
New sparse index: IntIndex
Indices: array([], dtype=int32)


Expected: [0. 0. 0. 0. 0.]
Actual: [1. 1. 1. 1. 1.]
Are they equal? False
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/25/repo.py", line 28, in <module>
    assert np.allclose(new_sparse.to_dense(), [0.0, 0.0, 0.0, 0.0, 0.0]), \
           ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Expected [0.0, 0.0, 0.0, 0.0, 0.0] but got [1. 1. 1. 1. 1.]
```
</details>

## Why This Is A Bug

This violates the fundamental principle that creating an array from another array should preserve the data values. The bug occurs because:

1. **Sparse storage optimization gone wrong**: In SparseArray, values equal to the fill_value are not explicitly stored (they're implicit). When all values equal the fill_value, the sparse index and sparse values arrays are empty.

2. **Incorrect reuse of sparse representation**: When constructing a SparseArray from another SparseArray with a different fill_value, the code incorrectly reuses the original sparse index and sparse values (lines 378-385 in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py`).

3. **Silent data corruption**: Because the sparse representation is reused without recalculation, all implicit values (those equal to the old fill_value) are now interpreted as the new fill_value, corrupting the data.

4. **Violates pandas documentation**: The pandas documentation for SparseArray indicates that data should be preserved when creating arrays from existing arrays. The constructor accepts a fill_value parameter when data is already a SparseArray, implying this operation should work correctly.

## Relevant Context

- **Pandas version tested**: 2.3.2
- **Code location**: `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py`, lines 376-385
- **Use case**: Users may legitimately want to change the fill_value to optimize storage (e.g., if the data distribution changes and a different value becomes more common)
- **Impact**: This bug causes complete data loss for any values that were equal to the original fill_value
- **Documentation**: [Pandas SparseArray documentation](https://pandas.pydata.org/docs/reference/api/pandas.arrays.SparseArray.html)

The bug is particularly insidious because:
- It fails silently without raising any warnings or errors
- The corrupted data looks plausible (uniform arrays are not uncommon)
- It only affects values equal to the original fill_value, making it data-dependent

## Proposed Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -374,14 +374,20 @@ class SparseArray(OpsMixin, ExtensionArray):
             fill_value = dtype.fill_value

         if isinstance(data, type(self)):
-            # disable normal inference on dtype, sparse_index, & fill_value
-            if sparse_index is None:
-                sparse_index = data.sp_index
-            if fill_value is None:
-                fill_value = data.fill_value
-            if dtype is None:
-                dtype = data.dtype
-            # TODO: make kind=None, and use data.kind?
-            data = data.sp_values
+            # If fill_value is changing, we need to convert to dense first
+            # to avoid data corruption when values equal to old fill_value
+            # would be misinterpreted as the new fill_value
+            if fill_value is not None and fill_value != data.fill_value:
+                data = data.to_dense()
+            else:
+                # disable normal inference on dtype, sparse_index, & fill_value
+                if sparse_index is None:
+                    sparse_index = data.sp_index
+                if fill_value is None:
+                    fill_value = data.fill_value
+                if dtype is None:
+                    dtype = data.dtype
+                # TODO: make kind=None, and use data.kind?
+                data = data.sp_values

         # Handle use-provided dtype
```