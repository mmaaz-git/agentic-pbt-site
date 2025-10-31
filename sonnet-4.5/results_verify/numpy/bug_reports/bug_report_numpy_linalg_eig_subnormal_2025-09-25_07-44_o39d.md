# Bug Report: numpy.linalg.eig Incorrect Eigenpairs for Subnormal Matrices

**Target**: `numpy.linalg.eig`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.linalg.eig` computes incorrect eigenpairs for matrices containing subnormal floating-point values, violating the fundamental eigenvalue property `A @ v = λ @ v`.

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
@settings(max_examples=300)
def test_eig_reconstruction(A):
    assume(np.all(np.isfinite(A)))

    try:
        eigenvalues, eigenvectors = np.linalg.eig(A)
        assume(np.all(np.isfinite(eigenvalues)) and np.all(np.isfinite(eigenvectors)))

        for i in range(len(eigenvalues)):
            Av = A @ eigenvectors[:, i]
            lambda_v = eigenvalues[i] * eigenvectors[:, i]
            assert np.allclose(Av, lambda_v, rtol=1e-6, atol=1e-9)
    except np.linalg.LinAlgError:
        assume(False)
```

**Failing input**: `array([[1.69764296e-127, 1.69764296e-127], [1.00000000e+000, 1.00000000e+000]])`

## Reproducing the Bug

```python
import numpy as np

A = np.array([[1.69764296e-127, 1.69764296e-127],
              [1.0, 1.0]])

eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:")
print(eigenvectors)
print()

v0 = eigenvectors[:, 0]
lam0 = eigenvalues[0]

Av = A @ v0
lam_v = lam0 * v0

print("Eigenpair 0 verification:")
print(f"  A @ v = {Av}")
print(f"  λ * v = {lam_v}")
print(f"  Match: {np.allclose(Av, lam_v)}")
print(f"  Error: {np.max(np.abs(Av - lam_v))}")
```

Output:
```
Eigenvalues: [1.69764296e-127 1.        ]
Eigenvectors:
[[1.00000000e+000 1.69764296e-127]
 [0.00000000e+000 1.00000000e+000]]

Eigenpair 0 verification:
  A @ v = [1.69764296e-127 1.        ]
  λ * v = [1.69764296e-127 0.        ]
  Match: False
  Error: 1.0
```

## Why This Is A Bug

The fundamental eigenvalue property states that for eigenpair (λ, v), the equation `A @ v = λ @ v` must hold. For the first eigenpair computed by `eig`, this property is violated with an error of magnitude 1.0.

The matrix is rank-deficient (both rows are linearly dependent), and the correct eigenvalues are 0 and 1. However, `eig` reports eigenvalue 1.69764296e-127 (a subnormal value from the matrix) instead of 0, with an incorrect eigenvector.

## Fix

This appears to be a numerical stability issue in the underlying eigenvalue algorithm (likely LAPACK) when handling matrices with subnormal values. Possible fixes:

1. Add input validation to detect and warn about matrices with subnormal values
2. Scale the matrix before computing eigenvalues, then scale results back
3. Document that `eig` may produce incorrect results for matrices with subnormal values
4. Investigate whether upgrading LAPACK or using different eigenvalue algorithms improves handling of subnormal values

A comprehensive fix would require investigating the underlying LAPACK routine used by numpy and potentially implementing preprocessing to handle subnormal matrices.