# Bug Report: numpy.linalg.eig Returns Incorrect Eigenvector

**Target**: `numpy.linalg.eig`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.linalg.eig` returns eigenvectors that violate the fundamental eigenvalue equation A @ v = λ * v for certain matrices, producing mathematically incorrect results.

## Property-Based Test

```python
import numpy as np
import numpy.linalg as la
from hypothesis import given, strategies as st, settings
from hypothesis.extra.numpy import arrays

@given(arrays(dtype=np.float64, shape=(3, 3), elements=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)))
@settings(max_examples=300)
def test_eig_property(a):
    try:
        result = la.eig(a)
        eigenvalues = result.eigenvalues
        eigenvectors = result.eigenvectors
    except la.LinAlgError:
        return

    for i in range(len(eigenvalues)):
        lam = eigenvalues[i]
        v = eigenvectors[:, i]

        lhs = a @ v
        rhs = lam * v

        if np.linalg.norm(v) > 1e-10:
            assert np.allclose(lhs, rhs, rtol=1e-4, atol=1e-7), f"A @ v != lambda * v for eigenpair {i}"
```

**Failing input**:
```python
a = np.array([[0.00000000e+00, 1.17549435e-38, 0.00000000e+00],
              [1.00000000e+00, 1.00000000e+00, 0.00000000e+00],
              [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]])
```

## Reproducing the Bug

```python
import numpy as np
import numpy.linalg as la

a = np.array([[0.00000000e+00, 1.17549435e-38, 0.00000000e+00],
              [1.00000000e+00, 1.00000000e+00, 0.00000000e+00],
              [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]])

result = la.eig(a)
eigenvalues = result.eigenvalues
eigenvectors = result.eigenvectors

lambda0 = eigenvalues[0]
v0 = eigenvectors[:, 0]

print(f"Eigenvalue: {lambda0}")
print(f"Eigenvector: {v0}")
print(f"A @ v = {a @ v0}")
print(f"lambda * v = {lambda0 * v0}")
print(f"Equal? {np.allclose(a @ v0, lambda0 * v0)}")
```

Output:
```
Eigenvalue: 0.0
Eigenvector: [1. 0. 0.]
A @ v = [0. 1. 0.]
lambda * v = [0. 0. 0.]
Equal? False
```

## Why This Is A Bug

The eigenvalue equation A @ v = λ * v is the defining property of eigenvalues and eigenvectors. For the reported eigenpair (λ=0, v=[1,0,0]), we have:

- A @ v = [0, 1, 0]
- λ * v = [0, 0, 0]

These are not equal, violating the fundamental mathematical definition. The eigenvector [1, 0, 0] is simply incorrect for eigenvalue 0. A correct eigenvector for eigenvalue 0 would be [0, 0, 1], which satisfies A @ [0,0,1] = [0,0,0] = 0 * [0,0,1].

## Fix

This appears to be a numerical stability issue in the underlying LAPACK eigenvalue computation when handling matrices with extreme value ranges (mixing values near machine epsilon with normal-scale values). The fix would require:

1. Improved numerical stability in the eigenvalue decomposition algorithm
2. Post-computation verification that returned eigenpairs satisfy A @ v ≈ λ * v
3. Better handling of matrices with extremely small off-diagonal elements

A proper fix would likely require changes to the underlying LAPACK calls or adding verification/refinement steps after the eigenvalue computation.