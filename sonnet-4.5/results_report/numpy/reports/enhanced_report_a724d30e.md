# Bug Report: pandas.core.arrays.sparse.SparseArray._concat_same_type Silently Loses Data When Concatenating Arrays with Different Fill Values

**Target**: `pandas.core.arrays.sparse.SparseArray._concat_same_type`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When concatenating SparseArrays with different fill values, the method silently loses data by only using the first array's fill value, causing values from subsequent arrays that match their own fill value to be replaced with the first array's fill value in the result.

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

if __name__ == "__main__":
    test_concat_preserves_values()
```

<details>

<summary>
**Failing input**: `vals1=[0, 0], vals2=[0, 0], fill1=1, fill2=0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/21/hypo.py", line 22, in <module>
    test_concat_preserves_values()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/21/hypo.py", line 6, in test_concat_preserves_values
    st.lists(st.integers(min_value=-10, max_value=10), min_size=2, max_size=10),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/21/hypo.py", line 19, in test_concat_preserves_values
    np.testing.assert_array_equal(result.to_dense(), expected)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/testing/_private/utils.py", line 1051, in assert_array_equal
    assert_array_compare(operator.__eq__, actual, desired, err_msg=err_msg,
    ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                         verbose=verbose, header='Arrays are not equal',
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                         strict=strict)
                         ^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/testing/_private/utils.py", line 916, in assert_array_compare
    raise AssertionError(msg)
AssertionError:
Arrays are not equal

Mismatched elements: 2 / 4 (50%)
Max absolute difference among violations: 1
Max relative difference among violations: inf
 ACTUAL: array([0, 0, 1, 1])
 DESIRED: array([0, 0, 0, 0])
Falsifying example: test_concat_preserves_values(
    # The test sometimes passed when commented parts were varied together.
    vals1=[0, 0],  # or any other generated value
    vals2=[0, 0],
    fill1=1,  # or any other generated value
    fill2=0,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/arrayprint.py:600
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py:3048
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/testing/_private/utils.py:862
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/testing/_private/utils.py:902
```
</details>

## Reproducing the Bug

```python
from pandas.arrays import SparseArray
import numpy as np

# Test case demonstrating the bug
arr1 = SparseArray([0, 0, 1], fill_value=0)
arr2 = SparseArray([2, 2, 3], fill_value=2)

result = SparseArray._concat_same_type([arr1, arr2])

print("Input arrays:")
print(f"  arr1: {list(arr1.to_dense())} (fill_value={arr1.fill_value})")
print(f"  arr2: {list(arr2.to_dense())} (fill_value={arr2.fill_value})")
print()
print("Concatenation result:")
print(f"  Result: {list(result.to_dense())} (fill_value={result.fill_value})")
print()
print("Expected result:")
print(f"  Expected: [0, 0, 1, 2, 2, 3]")
print()
print("Analysis:")
expected = np.concatenate([arr1.to_dense(), arr2.to_dense()])
actual = result.to_dense()
if not np.array_equal(expected, actual):
    print(f"  ERROR: Data loss detected!")
    print(f"  Missing values: {[e for e, a in zip(expected, actual) if e != a]}")
else:
    print("  OK: Data preserved correctly")
```

<details>

<summary>
ERROR: Data loss detected - values [2, 2] are replaced with [0, 0]
</summary>
```
Input arrays:
  arr1: [np.int64(0), np.int64(0), np.int64(1)] (fill_value=0)
  arr2: [np.int64(2), np.int64(2), np.int64(3)] (fill_value=2)

Concatenation result:
  Result: [np.int64(0), np.int64(0), np.int64(1), np.int64(0), np.int64(0), np.int64(3)] (fill_value=0)

Expected result:
  Expected: [0, 0, 1, 2, 2, 3]

Analysis:
  ERROR: Data loss detected!
  Missing values: [np.int64(2), np.int64(2)]
```
</details>

## Why This Is A Bug

This violates the fundamental expectation that concatenation preserves all data values. The bug occurs because `_concat_same_type` uses only the first array's fill_value for the result, but sparse arrays only store non-fill values explicitly. When arr2 has values equal to its fill_value (2), these are not stored in arr2.sp_values. During concatenation, these positions get filled with arr1's fill_value (0) instead, causing silent data corruption. This breaks the invariant that `concat([a, b]).to_dense() == np.concatenate([a.to_dense(), b.to_dense()])`. The method provides no documentation warning about this behavior, and users have no way to know data loss will occur without examining the source code.

## Relevant Context

The bug is in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py` at line 1183 where only the first array's fill_value is used: `fill_value = to_concat[0].fill_value`. The method then concatenates only the sparse values (sp_values) from each array without considering that arrays with different fill values need special handling.

SparseArrays store data efficiently by only keeping non-fill values in sp_values and their indices in sp_index. When fill values differ between arrays, values equal to each array's own fill_value are missing from sp_values but should still be preserved in the concatenated result.

This affects any code using pandas SparseArrays for memory-efficient data storage when different arrays use different fill values (e.g., combining datasets with different missing value indicators).

## Proposed Fix

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