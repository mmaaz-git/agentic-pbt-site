# Bug Report: numpy.linalg.eig Returns Invalid Eigenvectors

**Target**: `numpy.linalg.eig`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.linalg.eig` returns eigenvector/eigenvalue pairs that violate the fundamental eigenvalue equation `A @ v = λ * v` for certain matrices with very small (but non-zero) off-diagonal elements. The error can be as large as 1.0, making the eigenvector completely invalid.

## Property-Based Test

```python
import numpy as np
import numpy.linalg as LA
from hypothesis import given, strategies as st, assume, settings
from hypothesis.extra import numpy as npst

def matrices(min_side=2, max_side=5, dtype=np.float64):
    return st.integers(min_value=min_side, max_value=max_side).flatmap(
        lambda n: npst.arrays(
            dtype=dtype,
            shape=(n, n),
            elements=st.floats(
                min_value=-10.0,
                max_value=10.0,
                allow_nan=False,
                allow_infinity=False
            )
        )
    )

@given(matrices(min_side=2, max_side=4))
@settings(max_examples=50)
def test_eig_property(A):
    try:
        result = LA.eig(A)
        eigenvalues = result.eigenvalues
        eigenvectors = result.eigenvectors

        for i in range(len(eigenvalues)):
            v = eigenvectors[:, i]
            lam = eigenvalues[i]
            Av = A @ v
            lam_v = lam * v

            assert np.allclose(Av, lam_v, rtol=1e-4, atol=1e-6), \
                f"A @ v != λ * v for eigenvalue {i}. Max diff: {np.max(np.abs(Av - lam_v))}"
    except LA.LinAlgError:
        assume(False)
```

**Failing input**:
```python
A = np.array([[0.00000000e+000, 1.52474291e-300],
              [1.00000000e+000, 1.00000000e+000]])
```

## Reproducing the Bug

```python
import numpy as np
import numpy.linalg as LA

A = np.array([[0.0, 1.52474291e-300],
              [1.0, 1.0]])

result = LA.eig(A)
eigenvalues = result.eigenvalues
eigenvectors = result.eigenvectors

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:")
print(eigenvectors)

for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]

    Av = A @ v
    lam_v = lam * v

    print(f"\nEigenvalue {i}: λ = {lam}")
    print(f"  A @ v = {Av}")
    print(f"  λ * v = {lam_v}")
    print(f"  Error: ||A @ v - λ * v|| = {np.linalg.norm(Av - lam_v):.15e}")
    print(f"  Valid? {np.allclose(Av, lam_v, rtol=1e-5, atol=1e-7)}")
```

**Output:**
```
Eigenvalues: [0. 1.]
Eigenvectors:
[[1.00000000e+000 1.52474291e-300]
 [0.00000000e+000 1.00000000e+000]]

Eigenvalue 0: λ = 0.0
  A @ v = [0. 1.]
  λ * v = [0. 0.]
  Error: ||A @ v - λ * v|| = 1.000000000000000e+00
  Valid? False

Eigenvalue 1: λ = 1.0
  A @ v = [1.52474291e-300 1.00000000e+000]
  λ * v = [1.52474291e-300 1.00000000e+000]
  Error: ||A @ v - λ * v|| = 0.000000000000000e+00
  Valid? True
```

## Why This Is A Bug

The `numpy.linalg.eig` function's docstring explicitly states:

> "The normalized (unit "length") eigenvectors, such that the column `eigenvectors[:,i]` is the eigenvector corresponding to the eigenvalue `eigenvalues[i]`."

By definition, an eigenvector `v` corresponding to eigenvalue `λ` must satisfy `A @ v = λ * v`.

In this case, for eigenvalue `λ = 0.0`, the returned eigenvector is `v = [1, 0]`, but:
- `A @ v = [0, 1]`
- `λ * v = [0, 0]`

These are completely different! The error magnitude is 1.0, which is not a small numerical precision issue.

The bug occurs for matrices with very small (but non-zero) values in certain positions. Additional failing cases include:
- `A = [[0.0, 1e-100], [1.0, 1.0]]` - same error
- `A = [[1e-200, 1e-200], [1.0, 1.0]]` - similar error

Interestingly, matrices with exact zeros (e.g., `[[0.0, 0.0], [1.0, 1.0]]`) work correctly, suggesting the bug is related to how `eig` handles very small but non-zero values during computation.

## Fix

This appears to be a numerical stability issue in the eigenvalue computation algorithm (likely LAPACK's `geev` routine). The issue is that:

1. When computing eigenvalues via the characteristic polynomial, floating point arithmetic causes very small perturbations to round to exactly 0
2. The algorithm then tries to compute an eigenvector for λ=0
3. However, since the true eigenvalue is not exactly 0, the computed eigenvector doesn't satisfy the eigenvalue equation

Possible fixes:

1. **Improve numerical stability**: Use higher precision or different algorithms for nearly-singular matrices
2. **Post-computation verification**: After computing eigenpairs, verify they satisfy `A @ v ≈ λ * v` and refine if needed
3. **Document the limitation**: If the bug cannot be fixed, document that `eig` may return invalid eigenpairs for matrices with very small values and recommend using `eigh` for symmetric matrices or `svd` for better numerical stability

A quick workaround for users would be to always verify eigenpairs after computation:

```python
result = LA.eig(A)
for i in range(len(result.eigenvalues)):
    v = result.eigenvectors[:, i]
    lam = result.eigenvalues[i]
    if not np.allclose(A @ v, lam * v, rtol=1e-5, atol=1e-7):
        print(f"Warning: eigenpair {i} may be invalid")
```