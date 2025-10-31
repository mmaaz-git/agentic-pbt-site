# Bug Report: pandas_sparse_operation_2025-09-25 Invalid Sparse Representation After Operations

**Target**: `pandas.core.arrays.sparse.array._sparse_array_op`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Binary operations between SparseArrays can produce invalid sparse representations where `sp_values` contains fill values, violating the documented invariant that `sp_values` should only contain non-fill values.

## Property-Based Test

```python
from pandas.arrays import SparseArray
import numpy as np
from hypothesis import given, strategies as st, assume

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

# Run the test
if __name__ == "__main__":
    test_sparse_operation_invariant()
```

<details>

<summary>
**Failing input**: `values1=[1, 1, 1], values2=[0, 0, 1]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/5/hypo.py", line 23, in <module>
    test_sparse_operation_invariant()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/5/hypo.py", line 6, in test_sparse_operation_invariant
    st.lists(st.integers(min_value=1, max_value=10), min_size=3, max_size=10),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/5/hypo.py", line 18, in test_sparse_operation_invariant
    assert not np.any(result.sp_values == result.fill_value), \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: sp_values [1 1 0] contains fill_value 0
Falsifying example: test_sparse_operation_invariant(
    values1=[1, 1, 1],
    values2=[0, 0, 1],
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/5/hypo.py:19
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/arrayprint.py:1708
```
</details>

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
print(f"result.sp_index: {result.sp_index}")

# Check for the invariant violation
contains_fill = np.any(result.sp_values == result.fill_value)
print(f"\nInvariant violated (sp_values contains fill_value): {contains_fill}")
```

<details>

<summary>
Output shows sp_values contains fill_value 0
</summary>
```
result dense: [np.int64(0), np.int64(2), np.int64(2)]
result.sp_values: [0 2 2]
result.fill_value: 0
result.sp_index: IntIndex
Indices: array([0, 1, 2], dtype=int32)


Invariant violated (sp_values contains fill_value): True
```
</details>

## Why This Is A Bug

The pandas documentation for SparseArray explicitly states that `sp_values` contains the "non-fill_value values". Specifically:
- Line 298-299 in array.py: "Elements in data that are fill_value are not stored in the SparseArray"
- The entire concept of sparse arrays is based on not storing fill values to save memory
- When `sp_values` contains fill values, it violates this fundamental invariant

This bug occurs when one SparseArray in a binary operation has `ngaps == 0` (no gaps, meaning all positions have values). The code at lines 197-205 in `_sparse_array_op` converts both arrays to dense, performs the operation correctly, but then incorrectly reuses the original sparse index without filtering out positions where the result equals the fill value.

This creates an inconsistent sparse representation that could cause errors in operations that assume `sp_values` never contains fill values. While the dense representation remains correct, any code iterating over `sp_values` expecting only non-fill values could malfunction.

## Relevant Context

The bug is located in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py` in the `_sparse_array_op` function (lines 197-205). When either array has `ngaps == 0`, the function:
1. Converts both arrays to dense representation
2. Performs the operation on dense arrays
3. Reuses the sparse index from the array with no gaps
4. Creates a new SparseArray with the dense result but the old sparse index

The SparseArray constructor at lines 484-492 accepts a `sparse_index` parameter and when provided, directly uses the input data as `sp_values` without filtering, assuming the caller has already removed fill values.

Documentation reference: https://pandas.pydata.org/docs/reference/api/pandas.arrays.SparseArray.html

## Proposed Fix

After computing the dense result when `ngaps == 0`, properly re-sparsify the result to maintain the invariant:

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -197,12 +197,10 @@ def _sparse_array_op(
     if left.sp_index.ngaps == 0 or right.sp_index.ngaps == 0:
         with np.errstate(all="ignore"):
             result = op(left.to_dense(), right.to_dense())
             fill = op(_get_fill(left), _get_fill(right))

-        if left.sp_index.ngaps == 0:
-            index = left.sp_index
-        else:
-            index = right.sp_index
+        # Re-sparsify to maintain invariant that sp_values doesn't contain fill_value
+        fill_value = lib.item_from_zerodim(fill)
+        return SparseArray(result, fill_value=fill_value, kind=left.kind)
     elif left.sp_index.equals(right.sp_index):
         with np.errstate(all="ignore"):
```