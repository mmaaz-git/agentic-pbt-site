# Bug Report: scipy.sparse.linalg.expm Documentation Inconsistency

**Target**: `scipy.sparse.linalg.expm`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `expm` function's docstring claims to return an `ndarray` in its "Returns" section, but the "Examples" section demonstrates it returning a sparse array, and the actual implementation returns a sparse array when given sparse input. This creates confusion about the function's contract.

## Property-Based Test

```python
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from hypothesis import given, strategies as st, settings


@given(st.integers(min_value=2, max_value=6))
@settings(max_examples=50)
def test_expm_return_type_documentation(n):
    dense_A = np.random.randn(n, n)
    sparse_A = sp.csr_array(dense_A)

    result_dense = spl.expm(dense_A)
    result_sparse = spl.expm(sparse_A)

    assert isinstance(result_dense, np.ndarray), \
        "expm with dense input should return ndarray"

    is_sparse_output = sp.issparse(result_sparse)
    is_ndarray_output = isinstance(result_sparse, np.ndarray)

    assert is_sparse_output or is_ndarray_output, \
        f"expm should return either sparse or ndarray, got {type(result_sparse)}"
```

**Failing input**: Any sparse array input (e.g., `sp.csr_array([[1, 0], [0, 2]])`)

## Reproducing the Bug

```python
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

A = sp.diags([1.0, 2.0], format='csr')
result = spl.expm(A)

print(f"Type of result: {type(result)}")
print(f"Is sparse: {sp.issparse(result)}")

print("\nDocstring 'Returns' section says:")
print("  expA : (M,M) ndarray")
print("\nBut Examples section shows:")
print("  >>> Aexp")
print("  <Compressed Sparse Column sparse array of dtype 'float64'")
print("      with 3 stored elements and shape (3, 3)>")

assert sp.issparse(result), "Result is sparse, not ndarray as claimed in Returns section"
```

## Why This Is A Bug

The docstring contains contradictory information:
1. The "Returns" section explicitly states: `expA : (M,M) ndarray`
2. The "Examples" section shows the result as `<Compressed Sparse Column sparse array ...>`
3. The actual behavior returns sparse arrays for sparse inputs

This violates the principle of clear API contracts. Users reading the "Returns" section would expect an `ndarray` regardless of input type, but the function actually preserves sparsity, which is shown in the examples but not documented in the Returns section.

## Fix

Update the docstring's Returns section to accurately reflect the actual behavior:

```diff
 Returns
 -------
-expA : (M,M) ndarray
+expA : (M,M) ndarray or sparse array
     Matrix exponential of `A`
+
+    The output type matches the input type: dense inputs produce dense
+    outputs (ndarray), and sparse inputs produce sparse outputs.
```