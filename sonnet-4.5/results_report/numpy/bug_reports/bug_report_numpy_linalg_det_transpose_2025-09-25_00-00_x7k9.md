# Bug Report: numpy.linalg.det Determinant Sign Inconsistency with Transpose

**Target**: `numpy.linalg.det`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

numpy.linalg.det violates the fundamental mathematical property det(A) = det(A.T) for certain singular matrices, computing the same magnitude but opposite signs.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, settings, strategies as st
from hypothesis.extra import numpy as npst


def finite_float_matrix(shape):
    return npst.arrays(
        dtype=np.float64,
        shape=shape,
        elements=st.floats(
            min_value=-1e6,
            max_value=1e6,
            allow_nan=False,
            allow_infinity=False
        )
    )


@given(finite_float_matrix((3, 3)))
@settings(max_examples=300)
def test_det_transpose_invariant(A):
    det_A = np.linalg.det(A)
    det_AT = np.linalg.det(A.T)
    assert np.isclose(det_A, det_AT, rtol=1e-9, atol=1e-12)
```

**Failing input**:
```python
A = np.array([[     0. ,      0. , 416614.5],
              [416614.5, 416614.5, 416614.5],
              [416614.5, 416614.5, 416614.5]])
```

## Reproducing the Bug

```python
import numpy as np

A = np.array([[     0. ,      0. , 416614.5],
              [416614.5, 416614.5, 416614.5],
              [416614.5, 416614.5, 416614.5]])

det_A = np.linalg.det(A)
det_AT = np.linalg.det(A.T)

print(f"det(A)   = {det_A}")
print(f"det(A.T) = {det_AT}")
print(f"|det(A)|   = {abs(det_A)}")
print(f"|det(A.T)| = {abs(det_AT)}")
print(f"Signs match? {det_A * det_AT > 0}")
```

Output:
```
det(A)   = 10.102966428399663
det(A.T) = -10.102966428399663
|det(A)|   = 10.102966428399663
|det(A.T)| = 10.102966428399663
Signs match? False
```

## Why This Is A Bug

The mathematical property det(A) = det(A.T) holds for ALL square matrices in exact arithmetic. While singular matrices have determinant = 0, numerical error can produce small non-zero values. However, the sign of this numerical error should be CONSISTENT between A and A.T since they represent the same mathematical quantity.

The fact that magnitudes match exactly but signs differ indicates a systematic numerical stability issue in the determinant computation algorithm, likely related to how LU decomposition handles row vs column operations differently for near-singular matrices.

## Fix

This is a numerical stability issue in the underlying LAPACK routine. A potential mitigation would be to detect near-singular matrices (condition number > threshold) and return 0.0 consistently, but this would be a band-aid. The proper fix requires investigating the LU decomposition implementation in LAPACK's dgetrf routine to ensure symmetry with respect to transposition.

Since this is a deep numerical linear algebra issue, a comprehensive fix would require:
1. Identifying why the LU factorization produces different signs for A vs A.T
2. Either fixing the algorithm or documenting this as a known limitation
3. Adding tests to ensure det(A) = det(A.T) holds at least in sign for all matrices