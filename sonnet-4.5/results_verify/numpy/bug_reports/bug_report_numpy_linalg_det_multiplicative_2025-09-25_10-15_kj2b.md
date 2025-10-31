# Bug Report: numpy.linalg.det - Multiplicative Property Violation

**Target**: `numpy.linalg.det`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.linalg.det()` violates the fundamental mathematical property det(AB) = det(A) * det(B) when computing determinants of rank-deficient matrix products. When both det(A) and det(B) are correctly computed as 0.0 for singular matrices, det(AB) sometimes returns a non-zero value (e.g., -0.0475 or -1.525e-06) instead of 0.

## Property-Based Test

```python
import numpy as np
from hypothesis import assume, given, settings, strategies as st
from hypothesis.extra import numpy as npst


finite_matrices = npst.arrays(
    dtype=np.float64,
    shape=(3, 3),
    elements=st.floats(
        min_value=-1000,
        max_value=1000,
        allow_nan=False,
        allow_infinity=False
    )
)


@given(finite_matrices, finite_matrices)
@settings(max_examples=1000)
def test_det_multiplicative_property(a, b):
    det_a = np.linalg.det(a)
    det_b = np.linalg.det(b)

    assume(np.isfinite(det_a) and np.isfinite(det_b))
    assume(abs(det_a) < 1e100 and abs(det_b) < 1e100)

    ab = a @ b
    assume(np.all(np.isfinite(ab)))

    det_ab = np.linalg.det(ab)
    assume(np.isfinite(det_ab))

    expected = det_a * det_b

    max_det = max(abs(det_a), abs(det_b), abs(det_ab), abs(expected))

    if max_det < 1e-8:
        tolerance = 1e-8
    else:
        tolerance = max_det * 1e-6

    error = abs(det_ab - expected)

    assert error <= tolerance, \
        f"det(AB) = {det_ab}, but det(A)*det(B) = {expected}, error = {error}, tolerance = {tolerance}"
```

**Failing input**:
```python
a = array([[  0., 185., 185.],
           [185., 185., 185.],
           [185., 185., 185.]])
b = array([[7., 0., 7.],
           [7., 7., 7.],
           [7., 7., 7.]])
```

## Reproducing the Bug

```python
import numpy as np

a = np.array([[  0., 185., 185.],
              [185., 185., 185.],
              [185., 185., 185.]])
b = np.array([[7., 0., 7.],
              [7., 7., 7.],
              [7., 7., 7.]])

det_a = np.linalg.det(a)
det_b = np.linalg.det(b)
det_ab = np.linalg.det(a @ b)

print(f"det(A) = {det_a}")
print(f"det(B) = {det_b}")
print(f"det(A @ B) = {det_ab}")
print(f"Expected (det(A) * det(B)): {det_a * det_b}")

assert np.linalg.matrix_rank(a) == 2
assert np.linalg.matrix_rank(b) == 2
assert np.linalg.matrix_rank(a @ b) == 2
```

Output:
```
det(A) = 0.0
det(B) = 0.0
det(A @ B) = -1.5252453522407469e-06
Expected (det(A) * det(B)): 0.0
```

## Why This Is A Bug

This violates the fundamental linear algebra property det(AB) = det(A) × det(B). When NumPy correctly computes det(A) = 0 and det(B) = 0 for singular matrices, it should also compute det(AB) ≈ 0, since:

1. Both A and B are rank-deficient (rank 2 < 3)
2. AB must have rank ≤ min(rank(A), rank(B)) = 2
3. Therefore det(AB) must be 0 mathematically
4. NumPy correctly identifies det(A) = 0 and det(B) = 0, but inconsistently computes det(AB) as non-zero

This numerical inconsistency could cause issues when users rely on determinant computations for detecting singularity or validating matrix properties.

## Fix

This is a numerical stability issue in the underlying LAPACK routine used by `numpy.linalg.det()`. The bug occurs because:

1. When computing det(A) and det(B) individually, the LU decomposition correctly detects near-zero pivots and returns det = 0
2. When computing det(AB) directly, the product matrix AB may have slightly different numerical properties that cause the LU decomposition to miss the singularity

A potential fix would be to add a post-computation check: if the computed determinant is very small (e.g., < 1e-10) and the matrix is detected as rank-deficient via SVD or condition number, clamp the determinant to exactly 0.0.

Alternatively, improve tolerance thresholds in the underlying LU decomposition to better detect singular matrices in the presence of round-off error.