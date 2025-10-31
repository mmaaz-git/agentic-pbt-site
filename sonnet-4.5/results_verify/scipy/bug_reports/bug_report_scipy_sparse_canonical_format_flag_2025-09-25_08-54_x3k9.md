# Bug Report: scipy.sparse.coo_matrix has_canonical_format Flag Not Invalidated

**Target**: `scipy.sparse.coo_matrix.has_canonical_format`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `has_canonical_format` flag in `scipy.sparse.coo_matrix` is not invalidated when the underlying data arrays are modified directly, leading to a violated contract where the flag indicates canonical format but the actual data may not be canonical.

## Property-Based Test

```python
import numpy as np
import scipy.sparse as sp
from hypothesis import given, strategies as st, settings

@st.composite
def coo_matrices_with_duplicates(draw):
    n = draw(st.integers(min_value=2, max_value=15))
    size = draw(st.integers(min_value=2, max_value=30))
    data = draw(st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
                         min_size=size, max_size=size))
    row = draw(st.lists(st.integers(min_value=0, max_value=n-1), min_size=size, max_size=size))
    col = draw(st.lists(st.integers(min_value=0, max_value=n-1), min_size=size, max_size=size))
    return sp.coo_matrix((data, (row, col)), shape=(n, n))

@given(coo_matrices_with_duplicates())
@settings(max_examples=50)
def test_canonical_format_flag_invalidation(A):
    A.sum_duplicates()
    assert A.has_canonical_format

    original_data = A.data.copy()
    A.data[0] = A.data[0] + 1

    if A.has_canonical_format and not np.array_equal(A.data, original_data):
        raise AssertionError("BUG: has_canonical_format flag not invalidated after data modification")
```

**Failing input**: Any COO matrix after calling `sum_duplicates()`

## Reproducing the Bug

```python
import numpy as np
import scipy.sparse as sp

data = np.array([1.0, 2.0, 3.0])
row = np.array([0, 0, 0])
col = np.array([0, 0, 1])
A = sp.coo_matrix((data, (row, col)), shape=(2, 2))

A.sum_duplicates()
print(f"After sum_duplicates: has_canonical_format = {A.has_canonical_format}")

A.data[0] = 999.0
print(f"After data modification: has_canonical_format = {A.has_canonical_format}")
print(f"Data array: {A.data}")
```

Output:
```
After sum_duplicates: has_canonical_format = True
After data modification: has_canonical_format = True
Data array: [999.   3.]
```

## Why This Is A Bug

The `has_canonical_format` attribute is documented to indicate whether the COO matrix is in canonical format (no duplicate entries, sorted indices). When `sum_duplicates()` is called, this flag is set to `True` to indicate the matrix is now canonical.

However, scipy.sparse allows direct modification of the underlying `data`, `row`, and `col` arrays. When these arrays are modified, the `has_canonical_format` flag should be invalidated (set to `False`) because:

1. The canonical format guarantee is now violated
2. Code that relies on this flag (e.g., for optimization) will make incorrect assumptions
3. This violates the API contract that the flag accurately reflects the matrix state

This is a **contract violation bug** because the flag provides false information about the matrix state.

## Fix

There are two possible fixes:

**Option 1: Make arrays read-only after setting canonical format** (Recommended)

```diff
--- a/scipy/sparse/_coo.py
+++ b/scipy/sparse/_coo.py
@@ -200,6 +200,9 @@ class coo_matrix(_data_matrix, _minmax_mixin):
             self.row = self.row[unique_mask]
             self.col = self.col[unique_mask]
             self.has_canonical_format = True
+            self.data.flags.writeable = False
+            self.row.flags.writeable = False
+            self.col.flags.writeable = False
```

**Option 2: Use property setters to invalidate the flag**

```diff
--- a/scipy/sparse/_coo.py
+++ b/scipy/sparse/_coo.py
@@ -100,7 +100,18 @@ class coo_matrix(_data_matrix, _minmax_mixin):
     def __init__(self, arg1, shape=None, dtype=None, copy=False):
         # ... existing code ...
-        self.data = data
-        self.row = row
-        self.col = col
+        self._data = data
+        self._row = row
+        self._col = col
+
+    @property
+    def data(self):
+        return self._data
+
+    @data.setter
+    def data(self, value):
+        self._data = value
+        self.has_canonical_format = False
+        self.has_sorted_indices = False
```

The first option is simpler and prevents the bug by making modification impossible after canonical format is established. The second option allows modifications but correctly invalidates the flags.