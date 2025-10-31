# Bug Report: scipy.sparse.dia_array Matrix Multiplication Crash on Zero Matrices

**Target**: `scipy.sparse.dia_array @ scipy.sparse.dia_array`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

Matrix multiplication between two DIA sparse arrays with no stored diagonals (zero matrices) causes a `RuntimeError: vector::_M_default_append` crash instead of returning the expected zero matrix result.

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

if __name__ == "__main__":
    test_matmul_matches_numpy()
```

<details>

<summary>
**Failing input**: Two 1x1 DIA arrays containing all zeros
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/61/hypo.py", line 53, in <module>
    test_matmul_matches_numpy()
    ~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/61/hypo.py", line 43, in test_matmul_matches_numpy
    @settings(max_examples=100)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/61/hypo.py", line 47, in test_matmul_matches_numpy
    sp_result = (A @ B).toarray()
                 ~~^~~
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/sparse/_base.py", line 908, in __matmul__
    return self._matmul_dispatch(other)
           ~~~~~~~~~~~~~~~~~~~~~^^^^^^^
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/sparse/_base.py", line 812, in _matmul_dispatch
    return self._matmul_sparse(other)
           ~~~~~~~~~~~~~~~~~~~^^^^^^^
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/sparse/_dia.py", line 316, in _matmul_sparse
    offsets, data = dia_matmat(*self.shape, *self.data.shape,
                    ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                               self.offsets, self.data,
                               ^^^^^^^^^^^^^^^^^^^^^^^^
                               other.shape[1], *other.data.shape,
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                               other.offsets, other.data)
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: vector::_M_default_append
Falsifying example: test_matmul_matches_numpy(
    matrices=(<DIAgonal sparse array of dtype 'float64'
     	with 0 stored elements (0 diagonals) and shape (1, 1)>,
     <DIAgonal sparse array of dtype 'float64'
     	with 0 stored elements (0 diagonals) and shape (1, 1)>),
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/.local/lib/python3.13/site-packages/scipy/sparse/_coo.py:431
```
</details>

## Reproducing the Bug

```python
import numpy as np
import scipy.sparse as sp

A = sp.dia_array(np.zeros((1, 1)))
B = sp.dia_array(np.zeros((1, 1)))

result = A @ B
```

<details>

<summary>
RuntimeError: vector::_M_default_append
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/61/repo.py", line 7, in <module>
    result = A @ B
             ~~^~~
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/sparse/_base.py", line 908, in __matmul__
    return self._matmul_dispatch(other)
           ~~~~~~~~~~~~~~~~~~~~~^^^^^^^
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/sparse/_base.py", line 812, in _matmul_dispatch
    return self._matmul_sparse(other)
           ~~~~~~~~~~~~~~~~~~~^^^^^^^
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/sparse/_dia.py", line 316, in _matmul_sparse
    offsets, data = dia_matmat(*self.shape, *self.data.shape,
                    ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                               self.offsets, self.data,
                               ^^^^^^^^^^^^^^^^^^^^^^^^
                               other.shape[1], *other.data.shape,
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                               other.offsets, other.data)
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: vector::_M_default_append
```
</details>

## Why This Is A Bug

This violates fundamental mathematical properties and consistency expectations:

1. **Mathematical correctness**: Multiplying any matrix by a zero matrix should yield a zero matrix. This is a well-defined mathematical operation with an unambiguous result.

2. **Consistency across formats**: The operation works correctly for all other scipy sparse formats (CSR, CSC, COO, LIL, DOK, BSR) and dense NumPy arrays. Only DIA format crashes.

3. **Silent construction allowed**: Zero DIA matrices can be created without any warning or error using `sp.dia_array(np.zeros((n,m)))`, implying they are valid objects that should be fully functional.

4. **Hard crash vs graceful handling**: The code crashes with a low-level C++ STL error (`vector::_M_default_append`) rather than returning the mathematically correct result or raising a meaningful Python exception.

5. **Common use case**: Zero matrices naturally arise in many contexts including matrix initialization, masked operations where all elements are masked out, and intermediate algorithm results.

## Relevant Context

Testing reveals that:
- The bug affects zero DIA matrices of **all sizes** (1x1, 2x2, 10x10, etc.)
- Non-zero DIA matrices multiply correctly
- The issue occurs when `self.offsets` and/or `other.offsets` are empty arrays (no diagonals stored)
- The C++ function `dia_matmat` from `_sparsetools` module cannot handle empty diagonal arrays
- A simple workaround exists: convert to CSR format before multiplication

The bug is located in `/scipy/sparse/_dia.py` in the `_matmul_sparse` method (lines 307-322). The method already handles the case where matrix dimensions are zero (line 313-314), but fails to handle the case where matrices have valid dimensions but no stored diagonals.

## Proposed Fix

```diff
--- a/scipy/sparse/_dia.py
+++ b/scipy/sparse/_dia.py
@@ -313,6 +313,11 @@ class _dia_base(_data_matrix):
         if 0 in self.shape or 0 in other.shape:
             return self._dia_container((self.shape[0], other.shape[1]))

+        # Handle case where either matrix has no stored diagonals (zero matrix)
+        # This prevents dia_matmat from crashing on empty arrays
+        if len(self.offsets) == 0 or len(other.offsets) == 0:
+            return self._dia_container((self.shape[0], other.shape[1]))
+
         offsets, data = dia_matmat(*self.shape, *self.data.shape,
                                    self.offsets, self.data,
                                    other.shape[1], *other.data.shape,
```