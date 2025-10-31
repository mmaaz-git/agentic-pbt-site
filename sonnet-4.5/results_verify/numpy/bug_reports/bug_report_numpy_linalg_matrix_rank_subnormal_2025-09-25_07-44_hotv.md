# Bug Report: numpy.linalg.matrix_rank Incorrect Rank for Subnormal Matrices

**Target**: `numpy.linalg.matrix_rank`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.linalg.matrix_rank` incorrectly reports full rank for singular matrices containing subnormal floating-point values due to broken default tolerance calculation.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, assume, settings
from hypothesis.extra import numpy as npst
from hypothesis import strategies as st

def square_matrices(min_side=1, max_side=10):
    side = st.integers(min_value=min_side, max_value=max_side)
    return side.flatmap(
        lambda n: npst.arrays(
            dtype=np.float64,
            shape=(n, n),
            elements=st.floats(
                min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
            ),
        )
    )

@given(square_matrices(min_side=2, max_side=8))
@settings(max_examples=1000)
def test_singular_matrix_has_reduced_rank(A):
    assume(np.all(np.isfinite(A)))
    det = np.linalg.det(A)
    rank = np.linalg.matrix_rank(A)

    if det == 0:
        assert rank < A.shape[0], f"Singular matrix (det=0) must have rank < {A.shape[0]}, got rank={rank}"
```

**Failing input**: `array([[0.00000000e+000, 2.22507386e-311], [2.22507386e-311, 2.22507386e-311]])`

## Reproducing the Bug

```python
import numpy as np

A = np.array([[0.0, 2.22507386e-311],
              [2.22507386e-311, 2.22507386e-311]])

det = np.linalg.det(A)
rank = np.linalg.matrix_rank(A)

print(f"Determinant: {det}")
print(f"Determinant == 0: {det == 0}")
print(f"matrix_rank: {rank}")
print(f"Expected rank: < 2 (since det=0)")

U, s, Vh = np.linalg.svd(A)
default_tol = s.max() * max(A.shape) * np.finfo(float).eps
print(f"\nSingular values: {s}")
print(f"Default tolerance: {default_tol}")
print(f"Number of singular values > default_tol: {np.sum(s > default_tol)}")
```

Output:
```
Determinant: -0.0
Determinant == 0: True
matrix_rank: 2
Expected rank: < 2 (since det=0)

Singular values: [3.60024513e-311 1.37517127e-311]
Default tolerance: 0.0
Number of singular values > default_tol: 2
```

## Why This Is A Bug

A matrix with determinant 0 is singular and must have rank less than n. However, `matrix_rank` reports full rank (2) for this singular matrix.

The root cause is that the default tolerance calculation `tol = s.max() * max(shape) * eps` evaluates to 0 when all matrix values are subnormal. Since both singular values are greater than 0, they're both counted as significant, leading to incorrect rank 2.

This violates the fundamental property that singular matrices (det=0) cannot have full rank.

## Fix

The default tolerance calculation should use a minimum threshold to handle subnormal values:

```diff
--- a/numpy/linalg/_linalg.py
+++ b/numpy/linalg/_linalg.py
@@ -matrix_rank
-    tol = S.max() * max(M.shape) * np.finfo(S.dtype).eps
+    tol = max(S.max() * max(M.shape) * np.finfo(S.dtype).eps, np.finfo(S.dtype).tiny)
```

Alternatively, use `np.finfo(S.dtype).tiny` (smallest positive normal number) as the minimum tolerance threshold.