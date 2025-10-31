# Bug Report: pandas.core.arrays.sparse.SparseArray.nonzero Returns Incorrect Indices for Nonzero Fill Values

**Target**: `pandas.core.arrays.sparse.SparseArray.nonzero`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `nonzero()` method returns incorrect indices when the sparse array has a nonzero fill value, returning only indices of non-fill values rather than indices of all nonzero values.

## Property-Based Test

```python
import numpy as np
import pandas.core.arrays.sparse as sparse
from hypothesis import given, strategies as st, settings

@given(
    st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=100),
    st.integers(min_value=-10, max_value=10)
)
@settings(max_examples=200)
def test_nonzero_equivalence(data, fill_value):
    """nonzero should match dense nonzero"""
    arr = sparse.SparseArray(data, fill_value=fill_value)

    sparse_nonzero = arr.nonzero()
    dense_nonzero = arr.to_dense().nonzero()

    for s, d in zip(sparse_nonzero, dense_nonzero):
        np.testing.assert_array_equal(
            s, d,
            err_msg=f"nonzero() mismatch for data={data}, fill_value={fill_value}"
        )

if __name__ == "__main__":
    test_nonzero_equivalence()
```

<details>

<summary>
**Failing input**: `data=[1], fill_value=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 24, in <module>
    test_nonzero_equivalence()
    ~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 6, in test_nonzero_equivalence
    st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=100),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 18, in test_nonzero_equivalence
    np.testing.assert_array_equal(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        s, d,
        ^^^^^
        err_msg=f"nonzero() mismatch for data={data}, fill_value={fill_value}"
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/testing/_private/utils.py", line 1051, in assert_array_equal
    assert_array_compare(operator.__eq__, actual, desired, err_msg=err_msg,
    ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                         verbose=verbose, header='Arrays are not equal',
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                         strict=strict)
                         ^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/testing/_private/utils.py", line 803, in assert_array_compare
    raise AssertionError(msg)
AssertionError:
Arrays are not equal
nonzero() mismatch for data=[1], fill_value=1
(shapes (0,), (1,) mismatch)
 ACTUAL: array([], dtype=int32)
 DESIRED: array([0])
Falsifying example: test_nonzero_equivalence(
    data=[1],
    fill_value=1,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py:564
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py:578
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py:594
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py:595
```
</details>

## Reproducing the Bug

```python
import pandas.core.arrays.sparse as sparse
import numpy as np

# Test case 1: Array [1] with fill_value=1
print("Test case 1: Array [1] with fill_value=1")
arr1 = sparse.SparseArray([1], fill_value=1)
print(f"SparseArray: {arr1}")
print(f"SparseArray.nonzero(): {arr1.nonzero()}")
print(f"Dense array.nonzero(): {arr1.to_dense().nonzero()}")
print()

# Test case 2: Array [1, 2, 3] with fill_value=2
print("Test case 2: Array [1, 2, 3] with fill_value=2")
arr2 = sparse.SparseArray([1, 2, 3], fill_value=2)
print(f"SparseArray: {arr2}")
print(f"SparseArray.nonzero(): {arr2.nonzero()}")
print(f"Dense array.nonzero(): {arr2.to_dense().nonzero()}")
print()

# Test case 3: More complex case to show the issue clearly
print("Test case 3: Array [0, 1, 2, 0, 2, 2, 0] with fill_value=2")
arr3 = sparse.SparseArray([0, 1, 2, 0, 2, 2, 0], fill_value=2)
print(f"SparseArray: {arr3}")
print(f"SparseArray.nonzero(): {arr3.nonzero()}")
print(f"Dense array.nonzero(): {arr3.to_dense().nonzero()}")
```

<details>

<summary>
Output showing incorrect nonzero() results for nonzero fill values
</summary>
```
Test case 1: Array [1] with fill_value=1
SparseArray: [1]
Fill: 1
IntIndex
Indices: array([], dtype=int32)

SparseArray.nonzero(): (array([], dtype=int32),)
Dense array.nonzero(): (array([0]),)

Test case 2: Array [1, 2, 3] with fill_value=2
SparseArray: [1, 2, 3]
Fill: 2
IntIndex
Indices: array([0, 2], dtype=int32)

SparseArray.nonzero(): (array([0, 2], dtype=int32),)
Dense array.nonzero(): (array([0, 1, 2]),)

Test case 3: Array [0, 1, 2, 0, 2, 2, 0] with fill_value=2
SparseArray: [0, 1, 2, 0, 2, 2, 0]
Fill: 2
IntIndex
Indices: array([0, 1, 3, 6], dtype=int32)

SparseArray.nonzero(): (array([1], dtype=int32),)
Dense array.nonzero(): (array([1, 2, 4, 5]),)
```
</details>

## Why This Is A Bug

The `nonzero()` method is expected to return indices of all nonzero elements in an array, following NumPy's semantics. However, the current implementation has a fundamental logic error: when `fill_value` is nonzero, it only checks the explicitly stored sparse values for being nonzero, completely ignoring positions that contain the fill value.

The bug manifests in three ways:

1. **Complete miss of all nonzero values**: When all values equal a nonzero fill value (e.g., `[1]` with `fill_value=1`), `nonzero()` returns an empty array instead of all indices.

2. **Partial miss of nonzero values**: When some values equal a nonzero fill value (e.g., `[1, 2, 3]` with `fill_value=2`), `nonzero()` misses index 1 where the value is 2 (the fill value).

3. **Incorrect result set**: The method returns indices of "non-fill" values rather than "nonzero" values, which are fundamentally different concepts when the fill value itself is nonzero.

This violates the expected NumPy-compatible behavior where `array.nonzero()` should return indices of all elements where `array[i] != 0`, regardless of whether that value is a fill value or explicitly stored.

## Relevant Context

The bug is located in `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py` at lines 1410-1414:

```python
def nonzero(self) -> tuple[npt.NDArray[np.int32]]:
    if self.fill_value == 0:
        return (self.sp_index.indices,)
    else:
        return (self.sp_index.indices[self.sp_values != 0],)
```

The issue is in the `else` branch: when `fill_value != 0`, the code filters `sp_index.indices` to only include positions where `sp_values != 0`. This logic is incorrect because:
- It only examines explicitly stored values (`sp_values`)
- It ignores all positions containing the fill value
- When the fill value is nonzero (e.g., 1, 2, -1), those positions should be included in the nonzero result

SparseArrays with nonzero fill values are valid and useful in practice, such as:
- Mostly-ones matrices (fill_value=1)
- Default-value arrays where the default is nonzero
- Sparse representations where the common value is not zero

Documentation: While the method lacks explicit documentation, the name `nonzero()` has well-established semantics from NumPy that users reasonably expect to be followed.

## Proposed Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1409,10 +1409,20 @@ class SparseArray(OpsMixin, ExtensionArray):

     def nonzero(self) -> tuple[npt.NDArray[np.int32]]:
-        if self.fill_value == 0:
-            return (self.sp_index.indices,)
-        else:
-            return (self.sp_index.indices[self.sp_values != 0],)
+        # Return indices of all nonzero values in the array
+        # Must handle both sparse values and fill values correctly
+
+        if self.fill_value == 0:
+            # Only sparse values can be nonzero
+            return (self.sp_index.indices,)
+        elif self.fill_value != 0:
+            # Fill value is nonzero, so most/all positions may be nonzero
+            # Need to check the full array to get correct indices
+            return self.to_dense().nonzero()
+        else:
+            # NaN fill value case - filter sparse values for nonzero
+            # (though this branch may be unreachable if NaN != 0 is False)
+            return (self.sp_index.indices[self.sp_values != 0],)

     # ------------------------------------------------------------------------
     # Reductions
```