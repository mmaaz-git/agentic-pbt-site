# Bug Report: numpy.linalg.eig Returns Incorrect Eigenvector

**Target**: `numpy.linalg.eig`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.linalg.eig` returns an eigenvector that violates the fundamental eigenvalue equation A @ v = λ * v for ill-conditioned matrices with extreme value ranges, producing mathematically incorrect results without raising an error.

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
            assert np.allclose(lhs, rhs, rtol=1e-4, atol=1e-7)
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
lambda0 = result.eigenvalues[0]
v0 = result.eigenvectors[:, 0]

av = a @ v0
lambdav = lambda0 * v0

print(f"Eigenvalue: {lambda0}")
print(f"Eigenvector: {v0}")
print(f"A @ v = {av}")
print(f"lambda * v = {lambdav}")
print(f"Equal? {np.allclose(av, lambdav)}")
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

The eigenvalue equation A @ v = λ * v is the mathematical definition of eigenvalues and eigenvectors. For the reported eigenpair (λ=0, v=[1,0,0]):

- A @ v = [0, 1, 0]
- λ * v = [0, 0, 0]

The error magnitude is 1.0 (a full unit vector), not a small numerical error. The eigenvector [1, 0, 0] is fundamentally incorrect for eigenvalue 0. This represents silent data corruption - the function returns results that appear valid but are mathematically wrong.

## Fix

The issue stems from numerical instability when the matrix contains values spanning extreme ranges (1.17e-38 mixed with 1.0). The underlying LAPACK eigenvalue solver produces incorrect eigenvectors in such cases. Potential fixes:

1. Add input validation to detect and reject matrices with extreme value ranges
2. Improve numerical stability in the eigenvalue algorithm for such matrices
3. Add post-computation verification that eigenpairs satisfy A @ v ≈ λ * v and raise an error if they don't
4. Document the limitation that matrices with extreme value ranges may produce incorrect results