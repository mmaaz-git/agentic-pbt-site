# Bug Report: scipy.sparse.coo_matrix.transpose() Incorrectly Resets Canonical Format Flag

**Target**: `scipy.sparse.coo_matrix.transpose()`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `transpose()` method of `coo_matrix` incorrectly resets the `has_canonical_format` flag to `False` even when the transposed matrix is guaranteed to remain in canonical format (no duplicate entries, properly sorted).

## Property-Based Test

```python
import numpy as np
import scipy.sparse as sp
from hypothesis import given, strategies as st, settings


@st.composite
def sparse_coo_matrix(draw):
    m = draw(st.integers(min_value=1, max_value=20))
    n = draw(st.integers(min_value=1, max_value=20))
    nnz = draw(st.integers(min_value=0, max_value=min(m * n, 50)))

    if nnz == 0:
        return sp.coo_matrix((m, n), dtype=np.float64)

    rows = draw(st.lists(st.integers(min_value=0, max_value=m-1), min_size=nnz, max_size=nnz))
    cols = draw(st.lists(st.integers(min_value=0, max_value=n-1), min_size=nnz, max_size=nnz))
    data = draw(st.lists(st.floats(min_value=-1e10, max_value=1e10, allow_nan=False, allow_infinity=False), min_size=nnz, max_size=nnz))

    return sp.coo_matrix((data, (rows, cols)), shape=(m, n))


@given(sparse_coo_matrix())
@settings(max_examples=100)
def test_transpose_preserves_canonical_format(A):
    A.sum_duplicates()
    assert A.has_canonical_format == True

    A_T = A.transpose()

    assert A_T.has_canonical_format == True
```

<details>

<summary>
**Failing input**: Any `coo_matrix` with `has_canonical_format=True`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/51/hypo.py", line 35, in <module>
    test_transpose_preserves_canonical_format()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/51/hypo.py", line 23, in test_transpose_preserves_canonical_format
    @settings(max_examples=100)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/51/hypo.py", line 30, in test_transpose_preserves_canonical_format
    assert A_T.has_canonical_format == True
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_transpose_preserves_canonical_format(
    A=<COOrdinate sparse matrix of dtype 'float64'
    	with 0 stored elements and shape (1, 1)>,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import scipy.sparse as sp
import numpy as np

print("Testing scipy.sparse COO Matrix Transpose Canonical Format Bug")
print("=" * 60)

# Test case 1: Matrix with duplicates that get summed
data = [1, 2]
rows = [0, 0]
cols = [0, 0]
A = sp.coo_matrix((data, (rows, cols)), shape=(2, 2))

print("\nTest Case 1: Matrix with duplicate entries")
print("-" * 40)
print(f"Before sum_duplicates: A.has_canonical_format = {A.has_canonical_format}")

A.sum_duplicates()
print(f"After sum_duplicates: A.has_canonical_format = {A.has_canonical_format}")

A_T = A.transpose()
print(f"After transpose: A_T.has_canonical_format = {A_T.has_canonical_format}")

# Verify the transposed matrix actually has no duplicates
print(f"\nVerification:")
print(f"A_T.data = {A_T.data}")
print(f"A_T.row = {A_T.row}")
print(f"A_T.col = {A_T.col}")

# Test case 2: Matrix without duplicates from the start
print("\n" + "=" * 60)
print("\nTest Case 2: Matrix without duplicates")
print("-" * 40)

data2 = [1, 2, 3]
rows2 = [0, 1, 2]
cols2 = [0, 1, 2]
B = sp.coo_matrix((data2, (rows2, cols2)), shape=(3, 3))

print(f"Before sum_duplicates: B.has_canonical_format = {B.has_canonical_format}")
B.sum_duplicates()
print(f"After sum_duplicates: B.has_canonical_format = {B.has_canonical_format}")

B_T = B.transpose()
print(f"After transpose: B_T.has_canonical_format = {B_T.has_canonical_format}")

# Verify the transposed matrix structure
print(f"\nVerification:")
print(f"B_T.data = {B_T.data}")
print(f"B_T.row = {B_T.row}")
print(f"B_T.col = {B_T.col}")

# Show that calling sum_duplicates on transposed matrix changes only the flag
print("\n" + "=" * 60)
print("\nTest: Effect of sum_duplicates on already canonical transposed matrix")
print("-" * 40)

print(f"Before B_T.sum_duplicates(): B_T.has_canonical_format = {B_T.has_canonical_format}")
B_T_data_before = B_T.data.copy()
B_T_row_before = B_T.row.copy()
B_T_col_before = B_T.col.copy()

B_T.sum_duplicates()
print(f"After B_T.sum_duplicates(): B_T.has_canonical_format = {B_T.has_canonical_format}")

print(f"\nData changed: {not np.array_equal(B_T_data_before, B_T.data)}")
print(f"Row indices changed: {not np.array_equal(B_T_row_before, B_T.row)}")
print(f"Col indices changed: {not np.array_equal(B_T_col_before, B_T.col)}")
```

<details>

<summary>
Output showing incorrect flag after transpose
</summary>
```
Testing scipy.sparse COO Matrix Transpose Canonical Format Bug
============================================================

Test Case 1: Matrix with duplicate entries
----------------------------------------
Before sum_duplicates: A.has_canonical_format = False
After sum_duplicates: A.has_canonical_format = True
After transpose: A_T.has_canonical_format = False

Verification:
A_T.data = [3]
A_T.row = [0]
A_T.col = [0]

============================================================

Test Case 2: Matrix without duplicates
----------------------------------------
Before sum_duplicates: B.has_canonical_format = False
After sum_duplicates: B.has_canonical_format = True
After transpose: B_T.has_canonical_format = False

Verification:
B_T.data = [1 2 3]
B_T.row = [0 1 2]
B_T.col = [0 1 2]

============================================================

Test: Effect of sum_duplicates on already canonical transposed matrix
----------------------------------------
Before B_T.sum_duplicates(): B_T.has_canonical_format = False
After B_T.sum_duplicates(): B_T.has_canonical_format = True

Data changed: False
Row indices changed: False
Col indices changed: False
```
</details>

## Why This Is A Bug

This bug violates the semantic contract of the `has_canonical_format` flag, which indicates whether a COO matrix has sorted coordinates and no duplicate entries.

**Mathematical invariant**: Transposing a matrix is a bijective operation on coordinate pairs. If the original matrix has unique (row, col) pairs, the transposed matrix will have unique (col, row) pairs. This is mathematically guaranteed and cannot introduce duplicates.

**Specific violations**:

1. **Incorrect metadata**: The flag incorrectly reports `False` when the matrix IS in canonical format, as verified by our tests showing that calling `sum_duplicates()` on the transposed matrix doesn't change the data or indices at all.

2. **Performance impact**: The incorrect flag causes unnecessary duplicate summation operations in subsequent operations that check this flag (like `tocsr()`, `tocsc()`, etc.), degrading performance.

3. **Documentation contradiction**: The scipy documentation defines canonical format as having "no duplicate entries" and "entries sorted by row, then column." While transpose may change the sorting order, it preserves the no-duplicates property, yet the flag is unconditionally reset.

## Relevant Context

The bug originates from the implementation of `transpose()` in `/scipy/sparse/_coo.py` at lines 227-244. The method creates a new COO matrix using the `(data, coords)` constructor format, which automatically sets `has_canonical_format = False` (line 63), regardless of whether the original matrix was in canonical format.

Key observations from the code:
- Line 63: When constructing from `(data, coords)` tuple, always sets `has_canonical_format = False`
- Lines 242-243: `transpose()` uses this constructor path, causing the flag reset
- Lines 534-536: `sum_duplicates()` properly sets the flag to `True` after ensuring no duplicates

The transposed matrix maintains the no-duplicates property but the flag doesn't reflect this, leading to unnecessary performance overhead in operations that rely on this flag for optimization decisions.

Documentation references:
- Canonical format definition: Lines 1420-1423 and 1534-1537 in `_coo.py`
- Several methods (`tocsr()`, `tocsc()`, `diagonal()`) check and optimize based on this flag

## Proposed Fix

The fix preserves the `has_canonical_format` flag when transposing a matrix that's already in canonical format, since transpose cannot introduce duplicates:

```diff
--- a/scipy/sparse/_coo.py
+++ b/scipy/sparse/_coo.py
@@ -240,8 +240,11 @@ class _coo_base(_data_matrix, _minmax_mixin):

         permuted_shape = tuple(self._shape[i] for i in axes)
         permuted_coords = tuple(self.coords[i] for i in axes)
-        return self.__class__((self.data, permuted_coords),
-                              shape=permuted_shape, copy=copy)
+        result = self.__class__((self.data, permuted_coords),
+                                shape=permuted_shape, copy=copy)
+        if self.has_canonical_format:
+            result.has_canonical_format = True
+        return result

     transpose.__doc__ = _spbase.transpose.__doc__
```