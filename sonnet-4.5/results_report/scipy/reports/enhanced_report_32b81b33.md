# Bug Report: scipy.sparse.eye_array Inconsistent Diagonal Offset Boundary Behavior

**Target**: `scipy.sparse.eye_array`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`scipy.sparse.eye_array` inconsistently handles out-of-bounds diagonal offsets: it returns an empty sparse matrix when `abs(k) == n` but raises a `ValueError` when `abs(k) > n`, violating API consistency expectations.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Property-based test for scipy.sparse.eye_array offset bounds behavior."""

from hypothesis import given, strategies as st, settings
import scipy.sparse as sp

@given(
    st.integers(min_value=1, max_value=20),
    st.integers(min_value=-30, max_value=30)
)
@settings(max_examples=200)
def test_eye_with_large_offset(n, k):
    """eye_array should handle all offsets consistently"""
    E = sp.eye_array(n, k=k, format='csr')

    if abs(k) >= n:
        assert E.nnz == 0
    else:
        expected_nnz = n - abs(k)
        assert E.nnz == expected_nnz

if __name__ == "__main__":
    test_eye_with_large_offset()
```

<details>

<summary>
**Failing input**: `n=1, k=2`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/20/hypo.py", line 23, in <module>
    test_eye_with_large_offset()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/20/hypo.py", line 8, in test_eye_with_large_offset
    st.integers(min_value=1, max_value=20),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/20/hypo.py", line 14, in test_eye_with_large_offset
    E = sp.eye_array(n, k=k, format='csr')
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/sparse/_construct.py", line 413, in eye_array
    return _eye(m, n, k, dtype, format)
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/sparse/_construct.py", line 450, in _eye
    return diags_sparse(data, offsets=[k], shape=(m, n), dtype=dtype).asformat(format)
           ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/sparse/_construct.py", line 215, in diags_array
    raise ValueError(f"Offset {offset} (index {j}) out of bounds")
ValueError: Offset 2 (index 0) out of bounds
Falsifying example: test_eye_with_large_offset(
    n=1,
    k=2,
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction of scipy.sparse.eye_array inconsistent offset bounds checking bug."""

import scipy.sparse as sp

# Test case 1: k = n (should work and return empty matrix)
print("Test 1: eye_array(3, k=3, format='csr')")
try:
    E1 = sp.eye_array(3, k=3, format='csr')
    print(f"  Success: nnz={E1.nnz}, shape={E1.shape}")
except ValueError as e:
    print(f"  Error: {e}")

# Test case 2: k = n+1 (currently fails with ValueError)
print("\nTest 2: eye_array(3, k=4, format='csr')")
try:
    E2 = sp.eye_array(3, k=4, format='csr')
    print(f"  Success: nnz={E2.nnz}, shape={E2.shape}")
except ValueError as e:
    print(f"  Error: {e}")

# Test case 3: k = -n (should work and return empty matrix)
print("\nTest 3: eye_array(3, k=-3, format='csr')")
try:
    E3 = sp.eye_array(3, k=-3, format='csr')
    print(f"  Success: nnz={E3.nnz}, shape={E3.shape}")
except ValueError as e:
    print(f"  Error: {e}")

# Test case 4: k = -(n+1) (currently fails with ValueError)
print("\nTest 4: eye_array(3, k=-4, format='csr')")
try:
    E4 = sp.eye_array(3, k=-4, format='csr')
    print(f"  Success: nnz={E4.nnz}, shape={E4.shape}")
except ValueError as e:
    print(f"  Error: {e}")

# Demonstrating the minimal failing case
print("\nMinimal failing case: eye_array(1, k=2, format='csr')")
try:
    E_min = sp.eye_array(1, k=2, format='csr')
    print(f"  Success: nnz={E_min.nnz}, shape={E_min.shape}")
except ValueError as e:
    print(f"  Error: {e}")
```

<details>

<summary>
Output shows inconsistent boundary behavior
</summary>
```
Test 1: eye_array(3, k=3, format='csr')
  Success: nnz=0, shape=(3, 3)

Test 2: eye_array(3, k=4, format='csr')
  Error: Offset 4 (index 0) out of bounds

Test 3: eye_array(3, k=-3, format='csr')
  Success: nnz=0, shape=(3, 3)

Test 4: eye_array(3, k=-4, format='csr')
  Error: Offset -4 (index 0) out of bounds

Minimal failing case: eye_array(1, k=2, format='csr')
  Error: Offset 2 (index 0) out of bounds
```
</details>

## Why This Is A Bug

This violates expected behavior because `scipy.sparse.eye_array` exhibits inconsistent boundary conditions. When `abs(k) == n`, the function successfully returns an empty sparse matrix (0 non-zero elements), but when `abs(k) > n`, it raises a `ValueError`. Both cases represent equally valid empty diagonals - there are no matrix elements on diagonals outside the matrix bounds.

The function's documentation states that k is the "Diagonal to place ones on" but does not document any restrictions on k values or mention that a ValueError could be raised for certain offsets. Users would reasonably expect consistent behavior for all out-of-bounds diagonals.

This inconsistency:
1. **Breaks the principle of least surprise** - adjacent k values (k=n vs k=n+1) have fundamentally different behaviors
2. **Differs from NumPy precedent** - `numpy.diag()` handles all offsets gracefully, returning empty arrays for out-of-bounds diagonals
3. **Makes generic code fragile** - code iterating through diagonal offsets will unexpectedly fail at the k=n+1 boundary
4. **Creates a discontinuous API** - the function behaves non-monotonically as k increases

## Relevant Context

The error originates in `scipy/sparse/_construct.py` in the `diags_array` function at lines 120-122:

```python
length = min(m + offset, n - offset, K)
if length < 0:
    raise ValueError(f"Offset {offset} (index {j}) out of bounds")
```

When `offset > n`, the expression `n - offset` becomes negative, making `length < 0` which triggers the ValueError. However, when `offset == n`, `length` equals 0 (not negative), so no error is raised and an empty diagonal is correctly handled.

The function `eye_array` calls `_eye` which in turn calls `diags_array` (or `diags_sparse` depending on format), passing the diagonal offset k. The documentation for `eye_array` (accessible via `help(scipy.sparse.eye_array)`) does not mention any bounds on the k parameter.

For comparison, NumPy's diagonal functions handle large offsets consistently - `np.diag(np.eye(3), k=4)` returns an empty array `[]` without error.

## Proposed Fix

```diff
--- a/scipy/sparse/_construct.py
+++ b/scipy/sparse/_construct.py
@@ -117,9 +117,11 @@ def diags_array(diagonals, /, *, offsets=0, shape=None, format=None, dtype=Non
     for j, diagonal in enumerate(diagonals):
         offset = offsets[j]
         k = max(0, offset)
-        length = min(m + offset, n - offset, K)
-        if length < 0:
-            raise ValueError(f"Offset {offset} (index {j}) out of bounds")
+        length = max(0, min(m + offset, n - offset, K))
+        if length == 0 and len(diagonal) > 1:
+            raise ValueError(
+                f"Diagonal length (index {j}: {len(diagonal)}) at offset {offset} "
+                f"does not agree with array size ({m}, {n}).")
         try:
             data_arr[j, k:k+length] = diagonal[...,:length]
         except ValueError as e:
```