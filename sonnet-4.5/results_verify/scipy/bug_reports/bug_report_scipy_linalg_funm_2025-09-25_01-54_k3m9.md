# Bug Report: scipy.linalg.funm Incorrect Result for Nilpotent Matrices

**Target**: `scipy.linalg.funm`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`scipy.linalg.funm(A, np.exp)` returns incorrect results for certain nilpotent matrices, specifically returning the identity matrix instead of the correct matrix exponential. This differs from the correct result computed by `scipy.linalg.expm`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from hypothesis.extra.numpy import arrays
import numpy as np
import scipy.linalg

def square_matrix_strategy(n):
    return arrays(dtype=np.float64, shape=(n, n),
                  elements=st.floats(min_value=-100, max_value=100,
                                   allow_nan=False, allow_infinity=False))

@given(square_matrix_strategy(3))
@settings(max_examples=200, deadline=None)
def test_funm_consistency(A):
    try:
        norm_A = np.linalg.norm(A)
        assume(norm_A < 10)

        exp_A_funm = scipy.linalg.funm(A, np.exp)
        exp_A_direct = scipy.linalg.expm(A)

        if not np.iscomplexobj(exp_A_funm):
            assert np.allclose(exp_A_funm, exp_A_direct, rtol=1e-3, atol=1e-5), \
                f"funm(A, exp) should equal expm(A)"
    except (np.linalg.LinAlgError, ValueError, scipy.linalg.LinAlgError, OverflowError):
        assume(False)
```

**Failing input**:
```python
A = np.array([[0., 1., 0.],
              [0., 0., 0.],
              [0., 0., 0.]])
```

## Reproducing the Bug

```python
import numpy as np
import scipy.linalg

A = np.array([[0., 1., 0.],
              [0., 0., 0.],
              [0., 0., 0.]])

funm_result = scipy.linalg.funm(A, np.exp)
expm_result = scipy.linalg.expm(A)

print("funm(A, exp) =")
print(funm_result)

print("\nexpm(A) =")
print(expm_result)

print("\nExpected:")
print(np.eye(3) + A)

assert not np.allclose(funm_result, expm_result), \
    f"funm returns identity, but expm correctly returns I + A"
```

**Output:**
```
funm(A, exp) =
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]

expm(A) =
[[1. 1. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]

Expected:
[[1. 1. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
```

## Why This Is A Bug

For the given matrix A, which is nilpotent (A² = 0), the matrix exponential can be computed as:

```
exp(A) = I + A + A²/2! + A³/3! + ...
       = I + A  (since A² = 0)
       = [[1, 1, 0],
          [0, 1, 0],
          [0, 0, 1]]
```

The function `scipy.linalg.expm(A)` correctly computes this as `[[1, 1, 0], [0, 1, 0], [0, 0, 1]]`.

However, `scipy.linalg.funm(A, np.exp)` incorrectly returns the identity matrix `[[1, 0, 0], [0, 1, 0], [0, 0, 1]]`, missing the off-diagonal term from A.

This violates the expected behavior that `funm(A, np.exp)` should compute the same result as `expm(A)` for the matrix exponential function. The scipy documentation for `funm` states it computes matrix functions using eigenvalue decomposition, which may fail for defective matrices (matrices with repeated eigenvalues and insufficient eigenvectors). The warning message "funm result may be inaccurate, approximate err = 1" is emitted, but the result is not just inaccurate—it's fundamentally wrong.

## Fix

The issue appears to be in how `funm` handles matrices with repeated zero eigenvalues (nilpotent matrices). The function relies on eigenvalue decomposition, which is known to be problematic for defective matrices.

A potential fix would be:

1. Add a check in `funm` to detect when the matrix is defective or nearly defective
2. In such cases, fall back to a more robust algorithm (e.g., Schur decomposition or series expansion)
3. Or, at minimum, raise a more informative error instead of silently returning an incorrect result

The current warning is insufficient because:
- It's easy to miss in production code
- The error magnitude (err = 1) doesn't convey that the entire off-diagonal structure is wrong
- Users expect `funm(A, exp)` to match `expm(A)` for the exponential function

For the specific case of the exponential function, users should be directed to use `expm` instead of `funm(..., np.exp)` when the matrix may be defective.