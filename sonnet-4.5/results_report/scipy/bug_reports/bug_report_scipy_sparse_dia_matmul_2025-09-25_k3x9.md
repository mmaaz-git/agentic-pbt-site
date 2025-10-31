# Bug Report: scipy.sparse DIA Matrix Multiplication Crash

**Target**: `scipy.sparse.dia_array @ scipy.sparse.dia_array`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

Matrix multiplication of empty DIA sparse arrays causes a `RuntimeError: vector::_M_default_append` crash instead of returning the expected zero matrix result.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from hypothesis.extra.numpy import arrays
import numpy as np
import scipy.sparse as sp


@st.composite
def matmul_compatible_matrices(draw, max_size=10):
    m = draw(st.integers(min_value=1, max_value=max_size))
    n = draw(st.integers(min_value=1, max_value=max_size))
    k = draw(st.integers(min_value=1, max_value=max_size))

    shape_a = (m, n)
    shape_b = (n, k)

    dense_a = draw(arrays(
        dtype=np.float64,
        shape=shape_a,
        elements=st.floats(
            min_value=-100, max_value=100,
            allow_nan=False, allow_infinity=False
        )
    ))
    dense_b = draw(arrays(
        dtype=np.float64,
        shape=shape_b,
        elements=st.floats(
            min_value=-100, max_value=100,
            allow_nan=False, allow_infinity=False
        )
    ))

    mask_a = draw(arrays(dtype=np.bool_, shape=shape_a, elements=st.booleans()))
    mask_b = draw(arrays(dtype=np.bool_, shape=shape_b, elements=st.booleans()))

    dense_a = dense_a * mask_a
    dense_b = dense_b * mask_b

    return sp.dia_array(dense_a), sp.dia_array(dense_b)


@given(matmul_compatible_matrices())
@settings(max_examples=100)
def test_matmul_matches_numpy(matrices):
    A, B = matrices

    sp_result = (A @ B).toarray()
    np_result = A.toarray() @ B.toarray()

    assert np.allclose(sp_result, np_result, rtol=1e-8, atol=1e-8)
```

**Failing input**: Two 1x1 DIA arrays containing all zeros

## Reproducing the Bug

```python
import numpy as np
import scipy.sparse as sp

A = sp.dia_array(np.zeros((1, 1)))
B = sp.dia_array(np.zeros((1, 1)))

result = A @ B
```

Output:
```
RuntimeError: vector::_M_default_append
```

## Why This Is A Bug

Matrix multiplication of two zero matrices should return a zero matrix, not crash. The operation works correctly for all other sparse formats (CSR, CSC, COO, etc.) and for dense NumPy arrays. The expected result is a 1x1 zero matrix, but instead the code crashes with a C++ STL error.

This violates the fundamental expectation that valid mathematical operations on valid inputs should not crash, and that all sparse formats should behave consistently.

## Fix

The bug is in `/scipy/sparse/_dia.py` in the `_matmul_sparse` method around line 316. The issue occurs when `dia_matmat` is called with empty diagonal data (0 diagonals). The function should handle this edge case by returning an empty result matrix.

A potential fix would be to add an early return when both matrices have no stored diagonals:

```diff
diff --git a/scipy/sparse/_dia.py b/scipy/sparse/_dia.py
index abc123..def456 100644
--- a/scipy/sparse/_dia.py
+++ b/scipy/sparse/_dia.py
@@ -308,6 +308,10 @@ class _dia_base(_minmax_mixin, _spbase):
     # If any dimension is zero, return empty array immediately.
     if 0 in self.shape or 0 in other.shape:
         return self._dia_container((self.shape[0], other.shape[1]))
+
+    # Handle case where both matrices have no diagonals stored
+    if len(self.offsets) == 0 or len(other.offsets) == 0:
+        return self._dia_container((self.shape[0], other.shape[1]))

     offsets, data = dia_matmat(*self.shape, *self.data.shape,
                                self.offsets, self.data,
```