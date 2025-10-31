# Bug Report: scipy.sparse.eye Offset Validation for Non-Square Matrices

**Target**: `scipy.sparse.eye`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`scipy.sparse.eye()` raises an unclear error when the diagonal offset `k` is beyond the valid range for non-square matrices, while `numpy.eye()` handles the same case gracefully by returning a zero matrix. This inconsistency violates user expectations and the API contract.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import scipy.sparse as sp
import numpy as np

@settings(max_examples=100)
@given(
    st.integers(min_value=1, max_value=20),
    st.integers(min_value=1, max_value=20),
    st.integers(min_value=-10, max_value=10)
)
def test_eye_with_diagonal_offset(m, n, k):
    max_k = max(m, n) - 1
    k_bounded = k % (max_k + 1) if max_k > 0 else 0

    I = sp.eye(m, n, k=k_bounded)
    dense = I.toarray()
    expected = np.eye(m, n, k=k_bounded)

    assert I.shape == (m, n)
    assert np.allclose(dense, expected)
```

**Failing input**: `m=3, n=1, k=2`

## Reproducing the Bug

```python
import scipy.sparse as sp
import numpy as np

sp.eye(3, 1, k=2)
```

**Output:**
```
ValueError: Offset 2 (index 0) out of bounds
```

**Expected behavior (matching numpy):**
```python
import numpy as np
print(np.eye(3, 1, k=2))
```

**Output:**
```
[[0.]
 [0.]
 [0.]]
```

## Why This Is A Bug

The `scipy.sparse.eye()` function is meant to be a sparse equivalent of `numpy.eye()`. However, when given a diagonal offset that places the diagonal entirely outside the matrix bounds, the two functions behave differently:

1. **numpy.eye()**: Returns a zero matrix (valid behavior - there are no elements on diagonal k)
2. **scipy.sparse.eye()**: Raises `ValueError: Offset X (index 0) out of bounds` from internal `diags_array` function

**Why this violates expected behavior:**

1. **API Inconsistency**: Users expect scipy.sparse functions to behave like their numpy equivalents
2. **Unclear Error**: The error message comes from an internal function (`diags_array`) and doesn't explain what went wrong
3. **No Documentation**: The `eye()` docstring doesn't specify valid ranges for `k` based on matrix shape
4. **Asymmetric Behavior**: Some large offsets work (e.g., `sp.eye(1, 3, k=2)` succeeds) while others fail (e.g., `sp.eye(3, 1, k=2)` fails)

**Additional examples:**

```python
sp.eye(3, 1, k=0)   # ✓ Works
sp.eye(3, 1, k=1)   # ✓ Works
sp.eye(3, 1, k=2)   # ✗ Fails

sp.eye(1, 3, k=0)   # ✓ Works
sp.eye(1, 3, k=1)   # ✓ Works
sp.eye(1, 3, k=2)   # ✓ Works

sp.eye(5, 2, k=3)   # ✗ Fails
```

The failure pattern: when `k >= n` for an m×n matrix where m > n, the function fails.

## Fix

The fix should make `scipy.sparse.eye()` behavior consistent with `numpy.eye()` by handling out-of-bounds offsets gracefully:

```diff
--- a/scipy/sparse/_construct.py
+++ b/scipy/sparse/_construct.py
@@ -447,6 +447,13 @@ def _eye(m, n=None, k=0, dtype=float, format=None, is_array=False):
     if n is None:
         n = m
     m, n = int(m), int(n)
+
+    # Check if diagonal is entirely out of bounds
+    if k >= n or k <= -m:
+        # Return zero matrix (matching numpy.eye behavior)
+        from scipy.sparse import csr_array, csr_matrix
+        cls = csr_array if is_array else csr_matrix
+        return cls((m, n), dtype=dtype).asformat(format)

     if k > 0:
         if n < k:
```

This fix ensures that when the diagonal offset places all diagonal elements outside the matrix bounds, a zero matrix is returned instead of raising an error, matching `numpy.eye()` behavior.