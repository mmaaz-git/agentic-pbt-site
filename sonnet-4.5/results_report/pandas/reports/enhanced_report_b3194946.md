# Bug Report: pandas.core.arrays.sparse.SparseArray.nonzero Returns Incorrect Indices for Nonzero Fill Values

**Target**: `pandas.core.arrays.sparse.SparseArray.nonzero`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `SparseArray.nonzero()` method incorrectly omits positions containing nonzero fill values, returning different results than the mathematically correct `to_dense().nonzero()`.

## Property-Based Test

```python
import numpy as np
from pandas.arrays import SparseArray
from hypothesis import given, strategies as st

@given(
    st.lists(st.integers(min_value=-10, max_value=10), min_size=1, max_size=20),
    st.integers(min_value=-10, max_value=10)
)
def test_nonzero_matches_dense(data, fill_value):
    arr = SparseArray(data, fill_value=fill_value)
    sparse_result = arr.nonzero()[0]
    dense_result = arr.to_dense().nonzero()[0]

    assert np.array_equal(sparse_result, dense_result), \
        f"sparse.nonzero() != to_dense().nonzero() for data={data}, fill_value={fill_value}"

if __name__ == "__main__":
    test_nonzero_matches_dense()
```

<details>

<summary>
**Failing input**: `SparseArray([1], fill_value=1)`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/24/hypo.py", line 18, in <module>
    test_nonzero_matches_dense()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/24/hypo.py", line 6, in test_nonzero_matches_dense
    st.lists(st.integers(min_value=-10, max_value=10), min_size=1, max_size=20),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/24/hypo.py", line 14, in test_nonzero_matches_dense
    assert np.array_equal(sparse_result, dense_result), \
           ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: sparse.nonzero() != to_dense().nonzero() for data=[1], fill_value=1
Falsifying example: test_nonzero_matches_dense(
    data=[1],
    fill_value=1,
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
from pandas.arrays import SparseArray

arr = SparseArray([0, 1, 2, 2], fill_value=2)

print(f"Array: {arr}")
print(f"Fill value: {arr.fill_value}")
print(f"to_dense(): {arr.to_dense()}")
print(f"Expected nonzero positions (from to_dense().nonzero()): {arr.to_dense().nonzero()[0]}")
print(f"Actual nonzero positions (from arr.nonzero()): {arr.nonzero()[0]}")

try:
    assert np.array_equal(arr.nonzero()[0], arr.to_dense().nonzero()[0])
    print("Test passed: sparse.nonzero() matches to_dense().nonzero()")
except AssertionError:
    print("AssertionError: sparse.nonzero() does not match to_dense().nonzero()")
```

<details>

<summary>
Output showing incorrect nonzero indices
</summary>
```
Array: [0, 1, 2, 2]
Fill: 2
IntIndex
Indices: array([0, 1], dtype=int32)

Fill value: 2
to_dense(): [0 1 2 2]
Expected nonzero positions (from to_dense().nonzero()): [1 2 3]
Actual nonzero positions (from arr.nonzero()): [1]
AssertionError: sparse.nonzero() does not match to_dense().nonzero()
```
</details>

## Why This Is A Bug

This violates the fundamental mathematical definition of `nonzero()`. According to NumPy's documentation, `nonzero()` should "Return the indices of the elements that are non-zero." The method name and its established behavior in NumPy arrays mean that it should return indices where elements are not equal to zero (≠ 0), not indices where elements are not equal to the fill value.

In the example `SparseArray([0, 1, 2, 2], fill_value=2)`:
- The dense representation is `[0, 1, 2, 2]`
- Mathematically, positions 1, 2, and 3 contain nonzero values (1, 2, and 2 respectively)
- The current implementation incorrectly returns only `[1]`, missing positions 2 and 3 which contain the nonzero fill value 2

The bug occurs because the current implementation confuses "non-fill" with "nonzero". When `fill_value != 0`, it only checks explicitly stored values (`sp_values`) for being nonzero, completely ignoring that all positions NOT in `sp_index.indices` are implicitly filled with the `fill_value`, which may itself be nonzero.

## Relevant Context

The current implementation in pandas/core/arrays/sparse/array.py (lines 1410-1414):
```python
def nonzero(self) -> tuple[npt.NDArray[np.int32]]:
    if self.fill_value == 0:
        return (self.sp_index.indices,)
    else:
        return (self.sp_index.indices[self.sp_values != 0],)
```

Key observations:
- The method has no docstring, leaving expected behavior undefined
- When `fill_value == 0`, it correctly returns all stored indices (since stored values are nonzero)
- When `fill_value != 0`, it fails to account for positions containing the fill value
- The method is used internally in pandas in 19 different files with 43 occurrences
- This affects ALL SparseArrays with nonzero fill values

Workaround: Users can use `arr.to_dense().nonzero()` to get correct results, though this defeats the purpose of using sparse arrays.

## Proposed Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1410,7 +1410,15 @@ class SparseArray(OpsMixin, PclOps, ExtensionArray):
     def nonzero(self) -> tuple[npt.NDArray[np.int32]]:
         if self.fill_value == 0:
             return (self.sp_index.indices,)
         else:
-            return (self.sp_index.indices[self.sp_values != 0],)
+            # Get indices of explicitly stored values that are nonzero
+            nonzero_stored = self.sp_index.indices[self.sp_values != 0]
+
+            # If fill_value is nonzero, all non-stored positions are also nonzero
+            if self.fill_value != 0:
+                all_indices = np.arange(len(self), dtype=np.int32)
+                fill_positions = np.setdiff1d(all_indices, self.sp_index.indices)
+                return (np.sort(np.concatenate([nonzero_stored, fill_positions])),)
+
+            return (nonzero_stored,)
```