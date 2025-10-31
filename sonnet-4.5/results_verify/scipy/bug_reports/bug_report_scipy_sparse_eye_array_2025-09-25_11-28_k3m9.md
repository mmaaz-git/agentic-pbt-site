# Bug Report: scipy.sparse.eye_array Inconsistent Offset Bounds Checking

**Target**: `scipy.sparse.eye_array`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`scipy.sparse.eye_array` has inconsistent behavior for out-of-bounds diagonal offsets: it successfully returns an empty matrix when `abs(k) == n` but raises a `ValueError` when `abs(k) > n`, despite both cases representing equally valid (empty) diagonal matrices.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import scipy.sparse as sp

@given(
    st.integers(min_value=1, max_value=20),
    st.integers(min_value=-30, max_value=30)
)
@settings(max_examples=200)
def test_eye_with_large_offset(n, k):
    """eye should handle all offsets consistently"""
    E = sp.eye_array(n, k=k, format='csr')

    if abs(k) >= n:
        assert E.nnz == 0
    else:
        expected_nnz = n - abs(k)
        assert E.nnz == expected_nnz
```

**Failing input**: `n=1, k=2` (also fails for `n=3, k=4`, etc.)

## Reproducing the Bug

```python
import scipy.sparse as sp

E1 = sp.eye_array(3, k=3, format='csr')
print(f"eye_array(3, k=3): nnz={E1.nnz}")

E2 = sp.eye_array(3, k=4, format='csr')
```

Output:
```
eye_array(3, k=3): nnz=0
ValueError: Offset 4 (index 0) out of bounds
```

Expected: Both should succeed (returning empty matrices) or both should fail.
Actual: `k=3` succeeds, `k=4` fails.

## Why This Is A Bug

This violates the principle of least surprise and creates an inconsistent API:

1. **Inconsistent boundary behavior**: `k=n` is accepted but `k=n+1` is rejected, despite both representing empty diagonals
2. **Differs from NumPy**: `numpy.diag([], k=large_k)` is valid for any k
3. **Breaks monotonicity**: As k increases, the behavior changes discontinuously from "success with empty result" to "error"
4. **Makes generic code fragile**: Code that works with `k=n` will unexpectedly fail with `k=n+1`

The function's docstring does not document any restriction on offset values, making this behavior particularly surprising.

## Fix

The issue is in `scipy/sparse/_construct.py` in the `diags_array` function (line 122-123):

```python
length = min(m + offset, n - offset, K)
if length < 0:
    raise ValueError(f"Offset {offset} (index {j}) out of bounds")
```

When `offset > n`, we get `n - offset < 0`, making `length < 0`, which triggers the error.

The fix is to allow empty diagonals:

```diff
--- a/scipy/sparse/_construct.py
+++ b/scipy/sparse/_construct.py
@@ -119,7 +119,7 @@ def diags_array(diagonals, /, *, offsets=0, shape=None, format=None, dtype=Non
         offset = offsets[j]
         k = max(0, offset)
-        length = min(m + offset, n - offset, K)
-        if length < 0:
-            raise ValueError(f"Offset {offset} (index {j}) out of bounds")
+        length = max(0, min(m + offset, n - offset, K))
+        if length == 0 and len(diagonal) > 0 and len(diagonal) != 1:
+            raise ValueError(
+                f"Diagonal length (index {j}: {len(diagonal)}) at offset {offset} does not agree with array size ({m}, {n}).")
         try:
```

This change:
1. Uses `max(0, ...)` to ensure `length >= 0`, allowing empty diagonals
2. Still validates that if a diagonal is provided, it has the correct length
3. Makes behavior consistent across all offset values