# Bug Report: scipy.sparse.linalg.expm Documentation Incorrect Return Type

**Target**: `scipy.sparse.linalg.expm`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `expm` function's docstring states it returns an `ndarray`, but it actually returns a sparse matrix when given a sparse matrix input.

## Property-Based Test

```python
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from hypothesis import given, strategies as st, settings

@given(n=st.integers(min_value=2, max_value=10))
@settings(max_examples=50, deadline=None)
def test_expm_return_type_matches_docs(n):
    A = sp.csr_matrix((n, n))
    result = spl.expm(A)

    assert isinstance(result, np.ndarray), \
        f"Documentation says expm returns ndarray, but got {type(result)}"
```

**Failing input**: Any sparse matrix (e.g., `sp.csr_matrix([[1, 0], [0, 2]])`)

## Reproducing the Bug

```python
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

A_sparse = sp.csr_matrix([[1, 0], [0, 2]])
result = spl.expm(A_sparse)

print(f"Input type: {type(A_sparse)}")
print(f"Output type: {type(result)}")
print(f"Is output ndarray? {isinstance(result, np.ndarray)}")
print(f"Is output sparse? {sp.issparse(result)}")
```

**Output:**
```
Input type: <class 'scipy.sparse._csr.csr_matrix'>
Output type: <class 'scipy.sparse._csr.csr_matrix'>
Is output ndarray? False
Is output sparse? True
```

## Why This Is A Bug

The docstring explicitly states:

```
Returns
-------
expA : (M,M) ndarray
    Matrix exponential of `A`
```

However, the function returns a sparse matrix when given a sparse matrix input. This is a documentation/contract violation - the API differs from its documentation.

While the actual behavior (preserving sparsity) is likely intentional and useful, the documentation should be updated to accurately reflect this behavior.

## Fix

```diff
--- a/scipy/sparse/linalg/_matfuncs.py
+++ b/scipy/sparse/linalg/_matfuncs.py
@@ -10,7 +10,7 @@ def expm(A):

     Returns
     -------
-    expA : (M,M) ndarray
+    expA : (M,M) ndarray or sparse array
         Matrix exponential of `A`
+        Returns a sparse array if `A` is sparse, otherwise returns an ndarray.

     Notes
```