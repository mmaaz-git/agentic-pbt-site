# Bug Report: pandas.core.arrays.sparse.SparseArray.nonzero() Returns Empty Array When All Values Equal Non-Zero fill_value

**Target**: `pandas.core.arrays.sparse.SparseArray.nonzero()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `nonzero()` method of `SparseArray` returns incorrect indices when the `fill_value` is non-zero. It only returns indices where explicitly stored sparse values are non-zero, completely ignoring positions that implicitly contain the non-zero fill_value.

## Property-Based Test

```python
import numpy as np
from pandas.core.arrays.sparse import SparseArray
from hypothesis import given, strategies as st, settings


@given(
    st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=50),
    st.integers(min_value=-100, max_value=100)
)
@settings(max_examples=100)
def test_nonzero_consistency(data, fill_value):
    sparse = SparseArray(data, fill_value=fill_value)
    dense = np.array(data)

    sparse_nonzero = sparse.nonzero()[0]
    dense_nonzero = dense.nonzero()[0]

    np.testing.assert_array_equal(sparse_nonzero, dense_nonzero)


if __name__ == "__main__":
    # Run the property-based test
    test_nonzero_consistency()
```

<details>

<summary>
**Failing input**: `data=[1], fill_value=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/45/hypo.py", line 23, in <module>
    test_nonzero_consistency()
    ~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/45/hypo.py", line 7, in test_nonzero_consistency
    st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=50),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/45/hypo.py", line 18, in test_nonzero_consistency
    np.testing.assert_array_equal(sparse_nonzero, dense_nonzero)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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

(shapes (0,), (1,) mismatch)
 ACTUAL: array([], dtype=int32)
 DESIRED: array([0])
Falsifying example: test_nonzero_consistency(
    data=[1],
    fill_value=1,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/testing/_private/utils.py:793
```
</details>

## Reproducing the Bug

```python
import numpy as np
from pandas.core.arrays.sparse import SparseArray

# Test Case 1: data=[1], fill_value=1 (from the initial report)
print("Test Case 1: data=[1], fill_value=1")
data = [1]
fill_value = 1
sparse = SparseArray(data, fill_value=fill_value)
dense = np.array(data)

print(f"Data: {data}")
print(f"Fill value: {fill_value}")
print(f"Sparse nonzero(): {sparse.nonzero()[0]}")
print(f"Dense nonzero():  {dense.nonzero()[0]}")
print(f"Expected: {dense.nonzero()[0]}")
print(f"Match: {np.array_equal(sparse.nonzero()[0], dense.nonzero()[0])}")
print()

# Test Case 2: data=[2, 2, 0, 2, 5], fill_value=2
print("Test Case 2: data=[2, 2, 0, 2, 5], fill_value=2")
data = [2, 2, 0, 2, 5]
fill_value = 2
sparse = SparseArray(data, fill_value=fill_value)
dense = np.array(data)

print(f"Data: {data}")
print(f"Fill value: {fill_value}")
print(f"Sparse nonzero(): {sparse.nonzero()[0]}")
print(f"Dense nonzero():  {dense.nonzero()[0]}")
print(f"Expected: {dense.nonzero()[0]}")
print(f"Match: {np.array_equal(sparse.nonzero()[0], dense.nonzero()[0])}")
print()

# Additional test case to demonstrate the issue
print("Test Case 3: data=[0, 1, 2, 0, 3], fill_value=0 (should work correctly)")
data = [0, 1, 2, 0, 3]
fill_value = 0
sparse = SparseArray(data, fill_value=fill_value)
dense = np.array(data)

print(f"Data: {data}")
print(f"Fill value: {fill_value}")
print(f"Sparse nonzero(): {sparse.nonzero()[0]}")
print(f"Dense nonzero():  {dense.nonzero()[0]}")
print(f"Expected: {dense.nonzero()[0]}")
print(f"Match: {np.array_equal(sparse.nonzero()[0], dense.nonzero()[0])}")

# The following lines will raise an assertion error for the failing cases
print("\n" + "="*60)
print("Running assertion checks (will fail for non-zero fill_value)...")
print("="*60 + "\n")

# This will fail
data = [1]
fill_value = 1
sparse = SparseArray(data, fill_value=fill_value)
dense = np.array(data)
assert np.array_equal(sparse.nonzero()[0], dense.nonzero()[0]), "Test Case 1 Failed!"
```

<details>

<summary>
AssertionError: Test Case 1 Failed - SparseArray.nonzero() returns empty array when all values equal non-zero fill_value
</summary>
```
Test Case 1: data=[1], fill_value=1
Data: [1]
Fill value: 1
Sparse nonzero(): []
Dense nonzero():  [0]
Expected: [0]
Match: False

Test Case 2: data=[2, 2, 0, 2, 5], fill_value=2
Data: [2, 2, 0, 2, 5]
Fill value: 2
Sparse nonzero(): [4]
Dense nonzero():  [0 1 3 4]
Expected: [0 1 3 4]
Match: False

Test Case 3: data=[0, 1, 2, 0, 3], fill_value=0 (should work correctly)
Data: [0, 1, 2, 0, 3]
Fill value: 0
Sparse nonzero(): [1 2 4]
Dense nonzero():  [1 2 4]
Expected: [1 2 4]
Match: True

============================================================
Running assertion checks (will fail for non-zero fill_value)...
============================================================

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/45/repo.py", line 58, in <module>
    assert np.array_equal(sparse.nonzero()[0], dense.nonzero()[0]), "Test Case 1 Failed!"
           ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Test Case 1 Failed!
```
</details>

## Why This Is A Bug

The `nonzero()` method violates NumPy compatibility, which is a core design principle of pandas sparse arrays. The current implementation at line 1410-1414 in `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py` has a fundamental logic error:

```python
def nonzero(self) -> tuple[npt.NDArray[np.int32]]:
    if self.fill_value == 0:
        return (self.sp_index.indices,)
    else:
        return (self.sp_index.indices[self.sp_values != 0],)
```

When `fill_value` is non-zero:
- The method only checks if explicitly stored sparse values (`sp_values`) are non-zero
- It completely ignores that positions not in `sp_index.indices` contain the `fill_value`
- If `fill_value != 0`, these positions should be included in the nonzero result

This breaks the fundamental contract of `nonzero()`: it should return ALL indices where values are non-zero, not just the subset that happens to be explicitly stored in the sparse representation.

## Relevant Context

The bug is particularly problematic because:
1. **Silent data corruption**: Returns wrong indices without any warning or error
2. **Violates NumPy compatibility**: pandas documentation states SparseArrays provide "nearly identical" behavior to dense arrays
3. **Affects data analysis**: Any code relying on `nonzero()` for sparse arrays with non-zero fill values will produce incorrect results
4. **Common in certain domains**: Non-zero fill values are used in applications where a default non-zero value is common (e.g., default ratings, baseline measurements)

The implementation correctly handles the case when `fill_value == 0` because then only the explicitly stored values need to be checked. But the else branch is fundamentally flawed in its logic.

Documentation reference: https://pandas.pydata.org/pandas-docs/stable/user_guide/sparse.html states that SparseArray is designed for "nearly identical" behavior with dense arrays.

## Proposed Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1410,7 +1410,18 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):
     def nonzero(self) -> tuple[npt.NDArray[np.int32]]:
         if self.fill_value == 0:
             return (self.sp_index.indices,)
         else:
-            return (self.sp_index.indices[self.sp_values != 0],)
+            # When fill_value is non-zero, we need to return:
+            # 1. Indices where sp_values != 0 (explicitly stored non-zeros)
+            # 2. All indices NOT in sp_index (which contain the non-zero fill_value)
+
+            # Get indices of explicitly stored values that are non-zero
+            explicit_nonzero = self.sp_index.indices[self.sp_values != 0]
+
+            # Get all indices that contain fill_value (i.e., not explicitly stored)
+            all_indices = np.arange(len(self), dtype=np.int32)
+            mask = np.ones(len(self), dtype=bool)
+            mask[self.sp_index.indices] = False
+            fill_indices = all_indices[mask]
+
+            # Combine and sort the indices
+            return (np.sort(np.concatenate([explicit_nonzero, fill_indices])),)
```