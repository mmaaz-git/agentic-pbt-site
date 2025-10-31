# Bug Report: dask.array.eye Chunking Bug for Non-Square Matrices

**Target**: `dask.array.eye`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`dask.array.eye` crashes with "Missing dependency" error when creating non-square matrices (N != M) with chunk size >= M.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import dask.array as da

@given(
    st.integers(min_value=2, max_value=10),
    st.integers(min_value=2, max_value=10),
    st.integers(min_value=-1, max_value=1)
)
@settings(max_examples=300, deadline=None)
def test_eye_diagonal_ones(N, M, k):
    """
    Property: eye creates identity matrix with ones on diagonal
    Evidence: eye creates matrix with 1s on main diagonal
    """
    assume(N > abs(k) and M > abs(k))

    arr = da.eye(N, chunks=3, M=M, k=k)
    computed = arr.compute()

    for i in range(N):
        for j in range(M):
            if j - i == k:
                assert computed[i, j] == 1.0
            else:
                assert computed[i, j] == 0.0
```

**Failing input**: `N=2, M=3, k=0, chunks=3`

## Reproducing the Bug

```python
import dask.array as da
import numpy as np

arr = da.eye(2, chunks=3, M=3, k=0)
print(f"Created array with shape {arr.shape}")
result = arr.compute()
```

**Output:**
```
Created array with shape (2, 3)
ValueError: Missing dependency ('eye-b9a630cc5f44363256f427428c37836c', 0, 1) for dependents {'finalize-hlgfinalizecompute-a9e6c2aaaa9345289c71a77a2e65f521'}
```

The bug occurs when:
- N != M (non-square matrix)
- chunks >= M

Works correctly when:
- N == M (square matrix)
- chunks < M

## Why This Is A Bug

1. The docstring explicitly supports non-square matrices via the `M` parameter
2. NumPy's `eye` function works correctly with the same parameters: `np.eye(2, M=3)` produces the expected (2, 3) array
3. The chunking parameter should not affect correctness, only performance
4. Dask claims NumPy compatibility for this function

## Fix

**File**: `/dask/array/creation.py`, line 624

**Current code:**
```python
def eye(N, chunks="auto", M=None, k=0, dtype=float):
    ...
    vchunks, hchunks = normalize_chunks(chunks, shape=(N, M), dtype=dtype)
    chunks = vchunks[0]

    ...
    return Array(dsk, name_eye, shape=(N, M), chunks=(chunks, chunks), dtype=dtype)
```

**Problem**: Line 624 uses `chunks=(chunks, chunks)`, where `chunks` is `vchunks[0]`. This incorrectly applies the vertical chunk size to both dimensions.

**Fix:**
```diff
-    return Array(dsk, name_eye, shape=(N, M), chunks=(chunks, chunks), dtype=dtype)
+    return Array(dsk, name_eye, shape=(N, M), chunks=(vchunks, hchunks), dtype=dtype)
```

This ensures that both vertical and horizontal chunks are properly specified, allowing the task graph to reference all created tasks correctly.