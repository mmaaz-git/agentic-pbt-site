# Bug Report: scipy.sparse.eye_array ValueError for out-of-bounds diagonals

**Target**: `scipy.sparse.eye_array`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`scipy.sparse.eye_array(n, k=k)` raises a `ValueError` when `abs(k) >= n`, while `numpy.eye(n, k=k)` handles the same inputs gracefully by returning a zero matrix. This inconsistency violates the documented expectation that scipy.sparse functions mirror numpy's behavior.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
import scipy.sparse as sp


@given(
    n=st.integers(min_value=1, max_value=20),
    k=st.integers(min_value=-5, max_value=5)
)
@settings(max_examples=200)
def test_eye_matches_dense(n, k):
    sparse_eye = sp.eye_array(n, k=k).toarray()
    dense_eye = np.eye(n, k=k)

    np.testing.assert_array_equal(
        sparse_eye,
        dense_eye,
        err_msg="eye_array doesn't match numpy.eye"
    )
```

**Failing input**: `n=1, k=2`

## Reproducing the Bug

```python
import numpy as np
import scipy.sparse as sp

np_result = np.eye(1, k=2)
print(f"numpy.eye(1, k=2) = {np_result}")

try:
    sp_result = sp.eye_array(1, k=2)
    print(f"scipy.sparse.eye_array(1, k=2) = {sp_result.toarray()}")
except ValueError as e:
    print(f"scipy.sparse.eye_array(1, k=2) raised ValueError: {e}")
```

**Output:**
```
numpy.eye(1, k=2) = [[0.]]
scipy.sparse.eye_array(1, k=2) raised ValueError: Offset 2 (index 0) out of bounds
```

## Why This Is A Bug

1. **API Inconsistency**: The scipy.sparse documentation states that sparse array functions should mirror their dense numpy equivalents. `scipy.sparse.eye_array` is documented as the sparse equivalent of `numpy.eye`, but it raises an error in cases where `numpy.eye` succeeds.

2. **Undocumented Behavior**: The docstring for `eye_array` says it creates "ones on the kth diagonal and zeros elsewhere" but doesn't mention that it will raise a ValueError if the diagonal is out of bounds. Users reasonably expect it to return an all-zero matrix (no diagonal) in such cases, matching numpy's behavior.

3. **Surprising User Experience**: Users migrating from dense to sparse matrices would be surprised when their code breaks on edge cases that numpy handles gracefully.

4. **Pattern of Failures**: The bug affects cases where the diagonal is out of bounds:
   - For square matrices: `k >= n` or `k <= -n`
     - `eye_array(1, k=2)` fails (numpy returns `[[0.]]`)
     - `eye_array(2, k=3)` fails (numpy returns `[[0., 0.], [0., 0.]]`)
     - `eye_array(3, k=4)` fails (numpy returns 3x3 zero matrix)
   - For rectangular matrices: `k > n-1` or `k < -m+1`
     - `eye_array(1, 3, k=-2)` fails (numpy returns `[[0., 0., 0.]]`)
     - `eye_array(2, 5, k=-3)` fails (numpy returns 2x5 zero matrix)

## Fix

The bug is in `/scipy/sparse/_construct.py` in the `_eye` function, which calls `diags_array` with offset validation that is too strict. The fix should allow out-of-bounds offsets and return an appropriately shaped zero matrix:

```diff
--- a/scipy/sparse/_construct.py
+++ b/scipy/sparse/_construct.py
@@ -447,6 +447,11 @@ def _eye(m, n, k, dtype, format):
     else:
         n = int(n)

+    # Handle out-of-bounds diagonals by returning zero matrix (match numpy.eye)
+    if k >= n or k <= -m:
+        from ._base import _spbase
+        return _spbase._eye_empty(m, n, dtype, format)
+
     data = ones(max(0, min(m + min(k, 0), n - max(k, 0))), dtype=dtype)
     return diags_sparse(data, offsets=[k], shape=(m, n), dtype=dtype).asformat(format)
```

Alternatively, the validation in `diags_array` could be relaxed to allow out-of-bounds offsets when constructing diagonal matrices, treating them as empty diagonals rather than errors.