# Bug Report: scipy.sparse.linalg.spbandwidth Crashes on Empty Sparse Matrices

**Target**: `scipy.sparse.linalg.spbandwidth`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `spbandwidth` function crashes with a `ValueError` when given a sparse matrix with no non-zero elements (zero matrix).

## Property-Based Test

```python
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from hypothesis import given, strategies as st, settings

@given(
    n=st.integers(min_value=1, max_value=20),
    density=st.floats(min_value=0.0, max_value=1.0),
)
@settings(max_examples=200, deadline=None)
def test_spbandwidth_no_crash(n, density):
    rng = np.random.RandomState(0)
    A = sp.random(n, n, density=density, format='csr', random_state=rng)

    below, above = spl.spbandwidth(A)

    assert isinstance(below, int) and isinstance(above, int)
    assert 0 <= below < n
    assert 0 <= above < n
```

**Failing input**: `n=1, density=0.0` (any sparse matrix with no non-zero elements)

## Reproducing the Bug

```python
import scipy.sparse as sp
import scipy.sparse.linalg as spl

A = sp.csr_matrix((3, 3))
below, above = spl.spbandwidth(A)
```

**Error:**
```
ValueError: zero-size array to reduction operation minimum which has no identity
```

## Why This Is A Bug

A zero matrix (sparse matrix with no non-zero elements) is a valid sparse matrix. The function should handle this case gracefully and return an appropriate bandwidth value (logically `(0, 0)`), rather than crashing.

The documentation states that the function returns "the lower and upper bandwidth of a 2D numeric array" without any restriction that the array must have non-zero elements.

## Fix

```diff
--- a/scipy/sparse/linalg/_dsolve/linsolve.py
+++ b/scipy/sparse/linalg/_dsolve/linsolve.py
@@ -879,4 +879,6 @@ def spbandwidth(A):
         gap = A.coords[1] - A.coords[0]
     elif A.format == "dok":
         gap = [(c - r) for r, c in A.keys()] + [0]
         return -min(gap), max(gap)
+    if len(gap) == 0:
+        return 0, 0
     return max(-np.min(gap).item(), 0), max(np.max(gap).item(), 0)
```