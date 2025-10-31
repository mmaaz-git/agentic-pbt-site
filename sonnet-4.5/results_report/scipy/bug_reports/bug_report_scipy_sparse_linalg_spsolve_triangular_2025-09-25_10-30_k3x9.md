# Bug Report: scipy.sparse.linalg.spsolve_triangular int64 indices crash

**Target**: `scipy.sparse.linalg.spsolve_triangular`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`spsolve_triangular` crashes with `TypeError` when given sparse matrices with `int64` index arrays, even though this is a valid sparse matrix format and the default for coordinate-based construction.

## Property-Based Test

```python
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as linalg
from hypothesis import given, strategies as st, settings


@st.composite
def triangular_system(draw, kind='lower'):
    n = draw(st.integers(min_value=2, max_value=8))
    rows, cols, data = [], [], []

    for i in range(n):
        data.append(draw(st.floats(min_value=0.1, max_value=10)))
        rows.append(i)
        cols.append(i)

    for i in range(n):
        for j in range(i):
            if kind == 'lower' and draw(st.booleans()):
                val = draw(st.floats(min_value=-10, max_value=10))
                rows.append(i)
                cols.append(j)
                data.append(val)

    A = sp.csr_array((data, (rows, cols)), shape=(n, n))
    b = np.array([draw(st.floats(min_value=-100, max_value=100)) for _ in range(n)])
    return A, b


@given(triangular_system(kind='lower'))
@settings(max_examples=100, deadline=None)
def test_spsolve_triangular_lower(system):
    A, b = system
    x = linalg.spsolve_triangular(A, b, lower=True)
    result = A @ x
    assert np.allclose(result, b)
```

**Failing input**: Any sparse matrix constructed from coordinate format with default int64 indices.

## Reproducing the Bug

```python
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as linalg

rows = [0, 1, 1]
cols = [0, 0, 1]
data = [1.0, 0.5, 2.0]
A = sp.csr_array((data, (rows, cols)), shape=(2, 2))
b = np.array([1.0, 2.0])

print(f"A.indices.dtype: {A.indices.dtype}")
print(f"A.indptr.dtype: {A.indptr.dtype}")

x = linalg.spsolve_triangular(A, b, lower=True)
```

**Output**:
```
A.indices.dtype: int64
A.indptr.dtype: int64
TypeError: row indices and column pointers must be of type cint
```

## Why This Is A Bug

1. Sparse matrices constructed from coordinate format (the natural way to build them programmatically) default to `int64` indices
2. The function crashes instead of either accepting `int64` or automatically converting
3. The error message mentions "cint" which is not a standard NumPy type, making it unclear how to fix
4. Other scipy.sparse functions accept both `int32` and `int64` indices without issue

## Fix

The function should automatically convert index arrays to the required type. In `/home/npc/.local/lib/python3.13/site-packages/scipy/sparse/linalg/_dsolve/linsolve.py`, around line 733:

```diff
def spsolve_triangular(A, b, lower=True, overwrite_A=False, overwrite_b=False):
    # ... existing code ...

+   # Ensure indices and indptr are int32
+   if A.indices.dtype != np.int32:
+       A = A.copy()
+       A.indices = A.indices.astype(np.int32)
+       A.indptr = A.indptr.astype(np.int32)
+
    x, info = _superlu.gstrs(trans,
                             A.nnz,
                             A.data,
                             A.indices,
                             A.indptr,
                             b,
                             csc_construct_func,
                             options=options)
```

Alternatively, fix the underlying `_superlu.gstrs` function to accept both `int32` and `int64` indices.