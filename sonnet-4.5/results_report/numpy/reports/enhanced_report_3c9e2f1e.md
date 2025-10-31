# Bug Report: numpy.linalg.det Transpose Property Violation for Singular Matrices

**Target**: `numpy.linalg.det`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The function numpy.linalg.det violates the mathematical invariant det(A) = det(A.T) for certain singular matrices, returning values with identical magnitudes but opposite signs.

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


if __name__ == "__main__":
    # Run the test
    test_det_transpose_invariant()
```

<details>

<summary>
**Failing input**: `array([[ 1.0e+00, -1.4e+04,  1.0e+00], [ 1.0e+00,  1.0e+00,  1.0e+00], [ 1.0e+00,  1.0e+00,  1.0e+00]])`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/30/hypo.py", line 29, in <module>
    test_det_transpose_invariant()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/30/hypo.py", line 20, in test_det_transpose_invariant
    @settings(max_examples=300)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/30/hypo.py", line 24, in test_det_transpose_invariant
    assert np.isclose(det_A, det_AT, rtol=1e-9, atol=1e-12)
           ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_det_transpose_invariant(
    A=array([[ 1.0e+00, -1.4e+04,  1.0e+00],
           [ 1.0e+00,  1.0e+00,  1.0e+00],
           [ 1.0e+00,  1.0e+00,  1.0e+00]]),
)
```
</details>

## Reproducing the Bug

```python
import numpy as np

# The specific matrix that demonstrates the bug
A = np.array([[     0. ,      0. , 416614.5],
              [416614.5, 416614.5, 416614.5],
              [416614.5, 416614.5, 416614.5]])

# Compute determinants
det_A = np.linalg.det(A)
det_AT = np.linalg.det(A.T)

print("Matrix A:")
print(A)
print("\nMatrix A.T:")
print(A.T)
print(f"\ndet(A)   = {det_A}")
print(f"det(A.T) = {det_AT}")
print(f"\n|det(A)|   = {abs(det_A)}")
print(f"|det(A.T)| = {abs(det_AT)}")
print(f"\nSigns match? {det_A * det_AT > 0}")
print(f"Values are equal? {np.isclose(det_A, det_AT, rtol=1e-9, atol=1e-12)}")

# Additional analysis
print(f"\nDifference: {det_A - det_AT}")
print(f"Relative difference: {abs(det_A - det_AT) / max(abs(det_A), abs(det_AT))}")

# Check matrix properties
print(f"\nMatrix rank: {np.linalg.matrix_rank(A)}")
print(f"Condition number: {np.linalg.cond(A)}")
```

<details>

<summary>
Determinant sign mismatch for singular matrix
</summary>
```
Matrix A:
[[     0.       0.  416614.5]
 [416614.5 416614.5 416614.5]
 [416614.5 416614.5 416614.5]]

Matrix A.T:
[[     0.  416614.5 416614.5]
 [     0.  416614.5 416614.5]
 [416614.5 416614.5 416614.5]]

det(A)   = 10.102966428399663
det(A.T) = -10.102966428399663

|det(A)|   = 10.102966428399663
|det(A.T)| = 10.102966428399663

Signs match? False
Values are equal? False

Difference: 20.205932856799325
Relative difference: 2.0

Matrix rank: 2
Condition number: 1.6401331799177664e+17
```
</details>

## Why This Is A Bug

This bug violates the fundamental mathematical property that det(A) = det(A.T), which is guaranteed to hold for all square matrices in exact arithmetic. The issue manifests specifically with singular or near-singular matrices that have:

1. **Mathematical expectation**: For a rank-deficient matrix (rank 2 in a 3x3 matrix), the determinant should be exactly 0.0. The transpose of any matrix preserves its determinant value.

2. **Actual behavior**: The function returns small non-zero values (due to floating-point errors in the LU decomposition) but with opposite signs for A and A.T. The magnitudes are identical (10.102966... vs -10.102966...), indicating this is not random numerical noise but a systematic issue.

3. **Documentation gap**: The NumPy documentation for `numpy.linalg.det` (found in `/home/npc/pbt/agentic-pbt/envs/numpy_env/lib/python3.13/site-packages/numpy/linalg/_linalg.py:2381-2426`) states that determinants are computed via LU factorization using LAPACK routines but does not warn about potential sign inconsistencies for singular matrices.

4. **Impact**: Algorithms that rely on determinant signs for orientation detection, volume calculations with signed values, or matrix classification could produce incorrect results.

## Relevant Context

The affected matrices follow a specific pattern where:
- The matrix is singular (rank-deficient)
- Rows or columns contain duplicate entries
- The condition number is extremely high (>1e17 in the test case)
- The matrix structure has zeros in specific positions that differ between A and A.T

The underlying implementation uses LAPACK's `dgetrf` routine for LU factorization. The sign discrepancy likely arises from different pivot selection when processing rows vs columns in the factorization, combined with accumulated floating-point errors in near-singular cases.

Reference to NumPy source: The determinant computation is implemented in `numpy/linalg/_linalg.py` lines 2381-2432, which delegates to `_umath_linalg.det` with signature 'd->d' for real matrices.

## Proposed Fix

Since this is a deep numerical stability issue in the LAPACK LU decomposition routine, a complete fix would require modifying the underlying algorithm. However, a practical mitigation within NumPy could be:

```diff
--- a/numpy/linalg/_linalg.py
+++ b/numpy/linalg/_linalg.py
@@ -2427,7 +2427,17 @@ def det(a):
     a = asarray(a)
     _assert_stacked_square(a)
     t, result_t = _commonType(a)
     signature = 'D->D' if isComplexType(t) else 'd->d'
+
+    # Check for near-singular matrices and handle specially
+    # to ensure det(A) = det(A.T) property is preserved
+    if not isComplexType(t):
+        cond = np.linalg.cond(a)
+        if cond > 1e15:  # Matrix is near-singular
+            # For near-singular matrices, return 0 to avoid sign inconsistencies
+            result_shape = a.shape[:-2]
+            return np.zeros(result_shape, dtype=result_t)
+
     r = _umath_linalg.det(a, signature=signature)
     return r
```

Alternative approach: Document this limitation clearly in the function docstring, warning users that for singular or near-singular matrices, the sign of the determinant may not be reliable due to numerical precision limitations.