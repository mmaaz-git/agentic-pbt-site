# Bug Report: scipy.sparse.linalg.expm Documentation Incorrectly States Return Type

**Target**: `scipy.sparse.linalg.expm`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `scipy.sparse.linalg.expm` function documentation states it returns an `ndarray`, but it actually returns a sparse array when given sparse input, contradicting both the stated return type and the provided example.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import scipy.sparse as sp
import scipy.sparse.linalg as sla

@given(st.integers(min_value=1, max_value=5), st.random_module())
@settings(max_examples=50)
def test_expm_return_type_matches_documentation(n, random):
    A_dense = np.random.rand(n, n) * 0.1
    A = sp.csc_array(A_dense)

    result = sla.expm(A)

    assert isinstance(result, np.ndarray), \
        f"Documentation says expm returns ndarray, but got {type(result)}"
```

**Failing input**: Any sparse input (e.g., `n=1`)

## Reproducing the Bug

```python
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

A_sparse = sp.csc_array([[1.0, 0.0], [0.0, 2.0]])
result = sla.expm(A_sparse)

print(f"Input type: {type(A_sparse).__name__}")
print(f"Output type: {type(result).__name__}")
print(f"Documentation says: 'Returns expA : (M,M) ndarray'")
```

Output:
```
Input type: csc_array
Output type: csc_array
Documentation says: 'Returns expA : (M,M) ndarray'
```

## Why This Is A Bug

The function's docstring contains contradictory information:

1. The "Returns" section states: `expA : (M,M) ndarray`
2. The "Examples" section shows the output as a sparse array:
   ```
   >>> Aexp
   <Compressed Sparse Column sparse array of dtype 'float64'
       with 3 stored elements and shape (3, 3)>
   ```
3. The actual behavior returns sparse arrays for sparse input and ndarrays for dense input

This creates confusion for users who rely on the documentation to understand the function's behavior.

## Fix

Update the documentation to accurately reflect the actual behavior:

```diff
--- a/scipy/sparse/linalg/_matfuncs.py
+++ b/scipy/sparse/linalg/_matfuncs.py
@@ -660,7 +660,8 @@ def expm(A):

 Returns
 -------
-expA : (M,M) ndarray
+expA : (M,M) sparse array or ndarray
+    Sparse array if `A` is sparse, ndarray if `A` is dense.
     Matrix exponential of `A`
```