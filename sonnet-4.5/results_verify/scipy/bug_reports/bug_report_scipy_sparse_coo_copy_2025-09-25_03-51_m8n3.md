# Bug Report: scipy.sparse COO Matrix copy() Doesn't Preserve has_canonical_format Flag

**Target**: `scipy.sparse.coo_matrix.copy()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When copying a COO matrix that has `has_canonical_format=True` (e.g., after calling `sum_duplicates()`), the copy has `has_canonical_format=False` even though it has the same data with no duplicate coordinates. This violates the expectation that `copy()` creates an identical matrix with the same internal state.

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

    rows[0] = rows[1]
    cols[0] = cols[1]

    shape = (100, 100)
    mat = sparse.coo_matrix((data, (rows, cols)), shape=shape)
    mat.sum_duplicates()

    copied = mat.copy()

    assert copied.has_canonical_format, \
        "copy() should preserve has_canonical_format flag"
```

**Failing input**: COO matrix with 2 duplicate entries: `data=[0.0, 0.0], rows=[0, 0], cols=[0, 0]`

## Reproducing the Bug

```python
import numpy as np
from scipy import sparse

data = [1.0, 2.0]
rows = [0, 0]
cols = [0, 0]
shape = (100, 100)

mat = sparse.coo_matrix((data, (rows, cols)), shape=shape)
print(f"Before sum_duplicates: {mat.has_canonical_format}")

mat.sum_duplicates()
print(f"After sum_duplicates: {mat.has_canonical_format}")

copied = mat.copy()
print(f"After copy(): {copied.has_canonical_format}")

print(f"\nData identical: {np.allclose(mat.toarray(), copied.toarray())}")
```

**Output:**
```
Before sum_duplicates: False
After sum_duplicates: True
After copy(): False

Data identical: True
```

**Expected:** `After copy(): True` because copy() should preserve all internal state flags.

## Why This Is A Bug

The `copy()` method is expected to create a duplicate of the matrix with identical internal state. The `has_canonical_format` flag is part of the matrix's state and should be preserved. This violates the principle of least surprise and can cause:

1. Performance issues - the copied matrix may be unnecessarily processed with `sum_duplicates()` again
2. State inconsistency - two matrices with identical data having different flags
3. API contract violation - copy() should preserve all object state

## Fix

The bug is in the `copy()` method of COO matrices. The fix is to preserve the `has_canonical_format` flag when creating a copy.

```diff
--- a/scipy/sparse/_coo.py
+++ b/scipy/sparse/_coo.py
@@ -XXX,X +XXX,X @@ class coo_matrix(_data_matrix):
     def copy(self):
         M, N = self.shape
         data = self.data.copy()
         row = self.row.copy()
         col = self.col.copy()
         A = coo_matrix(
             (data, (row, col)), shape=(M, N), dtype=self.dtype, copy=False
         )
+        A.has_canonical_format = self.has_canonical_format
         A.has_sorted_indices = self.has_sorted_indices
         return A
```

Note: The exact implementation may vary by scipy version. The key change is copying the `has_canonical_format` attribute from the source matrix to the copy, similar to how `has_sorted_indices` is already handled (if present in the implementation).