# Bug Report: scipy.sparse.coo_matrix.copy() Doesn't Preserve has_canonical_format Flag

**Target**: `scipy.sparse.coo_matrix.copy()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When copying a COO sparse matrix that has been canonicalized (has_canonical_format=True after calling sum_duplicates()), the copy() method fails to preserve the has_canonical_format flag, resulting in unnecessary performance overhead when operations assume the matrix needs duplicate checking.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
import numpy as np
from scipy import sparse

@given(
    data=st.lists(st.floats(allow_nan=False, allow_infinity=False,
                           min_value=-100, max_value=100),
                  min_size=2, max_size=100),
    rows=st.lists(st.integers(min_value=0, max_value=99),
                  min_size=2, max_size=100),
    cols=st.lists(st.integers(min_value=0, max_value=99),
                  min_size=2, max_size=100)
)
def test_canonical_format_preserved_after_copy(data, rows, cols):
    min_len = min(len(data), len(rows), len(cols))
    data = list(data[:min_len])
    rows = list(rows[:min_len])
    cols = list(cols[:min_len])

    assume(min_len >= 2)

    # Force duplicate coordinates
    rows[0] = rows[1]
    cols[0] = cols[1]

    shape = (100, 100)
    mat = sparse.coo_matrix((data, (rows, cols)), shape=shape)
    mat.sum_duplicates()

    copied = mat.copy()

    assert copied.has_canonical_format, \
        f"copy() should preserve has_canonical_format flag. Original: {mat.has_canonical_format}, Copy: {copied.has_canonical_format}"
```

<details>

<summary>
**Failing input**: `data=[0.0, 0.0], rows=[0, 0], cols=[0, 0]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/47/hypo.py", line 37, in <module>
    test_canonical_format_preserved_after_copy()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/47/hypo.py", line 6, in test_canonical_format_preserved_after_copy
    data=st.lists(st.floats(allow_nan=False, allow_infinity=False,
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/47/hypo.py", line 32, in test_canonical_format_preserved_after_copy
    assert copied.has_canonical_format, \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: copy() should preserve has_canonical_format flag. Original: True, Copy: False
Falsifying example: test_canonical_format_preserved_after_copy(
    # The test always failed when commented parts were varied together.
    data=[0.0, 0.0],  # or any other generated value
    rows=[0, 0],  # or any other generated value
    cols=[0, 0],  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
from scipy import sparse

# Create a COO matrix with duplicate coordinates
data = [1.0, 2.0]
rows = [0, 0]  # duplicate row index
cols = [0, 0]  # duplicate column index
shape = (100, 100)

# Create COO matrix
mat = sparse.coo_matrix((data, (rows, cols)), shape=shape)
print(f"Before sum_duplicates: has_canonical_format = {mat.has_canonical_format}")
print(f"Data shape before: {mat.data.shape}")

# Call sum_duplicates to merge duplicate entries
mat.sum_duplicates()
print(f"\nAfter sum_duplicates: has_canonical_format = {mat.has_canonical_format}")
print(f"Data shape after: {mat.data.shape}")
print(f"Data value: {mat.data}")

# Copy the matrix
copied = mat.copy()
print(f"\nAfter copy(): has_canonical_format = {copied.has_canonical_format}")
print(f"Copied data shape: {copied.data.shape}")
print(f"Copied data value: {copied.data}")

# Check if data is identical
print(f"\nData arrays equal: {np.array_equal(mat.data, copied.data)}")
print(f"Row indices equal: {np.array_equal(mat.row, copied.row)}")
print(f"Col indices equal: {np.array_equal(mat.col, copied.col)}")
print(f"Dense arrays equal: {np.allclose(mat.toarray(), copied.toarray())}")

# Bug demonstration: the canonical format flag is lost
print(f"\nBUG: Original has_canonical_format={mat.has_canonical_format}, Copy has_canonical_format={copied.has_canonical_format}")
print("Expected: Both should be True after copy()")
```

<details>

<summary>
Output demonstrating the bug
</summary>
```
Before sum_duplicates: has_canonical_format = False
Data shape before: (2,)

After sum_duplicates: has_canonical_format = True
Data shape after: (1,)
Data value: [3.]

After copy(): has_canonical_format = False
Copied data shape: (1,)
Copied data value: [3.]

Data arrays equal: True
Row indices equal: True
Col indices equal: True
Dense arrays equal: True

BUG: Original has_canonical_format=True, Copy has_canonical_format=False
Expected: Both should be True after copy()
```
</details>

## Why This Is A Bug

The `copy()` method is expected to create an identical duplicate of a matrix, including its internal state. The `has_canonical_format` flag is a critical piece of internal state that indicates whether the matrix has:
1. No duplicate coordinate entries (they've been summed)
2. Sorted coordinates by row, then column

When this flag is not preserved during copy:
- **Performance degradation**: Operations that check for duplicates will unnecessarily re-check the already canonical matrix, causing significant performance overhead
- **State inconsistency**: Two matrices with identical data have different internal state flags
- **Violates copy semantics**: The general expectation is that `copy()` creates an object that behaves identically to the original

The bug occurs because the `copy()` method in the parent `_data_matrix` class calls `_with_data()`, which in COO's implementation creates a new matrix without preserving the `has_canonical_format` attribute. The copied matrix defaults to `has_canonical_format = False` even though its data is already in canonical format.

## Relevant Context

The `has_canonical_format` flag serves as an important optimization in scipy.sparse COO matrices. When set to True, it allows methods to skip expensive duplicate-checking operations. Many operations (like `tocsr()`, `tocsc()`, `todia()`, `todok()`) call `sum_duplicates()` internally, which is a no-op when `has_canonical_format` is True but performs sorting and summation when False.

The flag is properly set in some COO matrix creation paths:
- Line 43 in `_coo.py`: Set to True when creating empty matrix
- Line 63: Set to False when creating from coordinate arrays
- Line 70: Preserved when copying from another sparse matrix with `copy=True`
- Line 96: Set to True when creating from dense array
- Line 536: Set to True after calling `sum_duplicates()`

However, the `_with_data()` method (lines 517-525) which is used by `copy()` doesn't preserve this flag.

## Proposed Fix

The fix is to modify the `_with_data()` method in the COO implementation to preserve the `has_canonical_format` flag:

```diff
--- a/scipy/sparse/_coo.py
+++ b/scipy/sparse/_coo.py
@@ -522,7 +522,9 @@ class _coo_base(_data_matrix, _minmax_mixin):
             coords = tuple(idx.copy() for idx in self.coords)
         else:
             coords = self.coords
-        return self.__class__((data, coords), shape=self.shape, dtype=data.dtype)
+        result = self.__class__((data, coords), shape=self.shape, dtype=data.dtype)
+        result.has_canonical_format = self.has_canonical_format
+        return result

     def sum_duplicates(self) -> None:
         """Eliminate duplicate entries by adding them together
```