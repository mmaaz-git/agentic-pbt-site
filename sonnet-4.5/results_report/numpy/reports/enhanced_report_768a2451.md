# Bug Report: numpy.linalg.eig Returns Mathematically Incorrect Eigenvector

**Target**: `numpy.linalg.eig`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.linalg.eig` returns eigenvectors that violate the fundamental eigenvalue equation A @ v = λ * v by a magnitude of 1.0 for certain matrices with extremely small off-diagonal elements near machine epsilon.

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

# Run the test to find a failure
if __name__ == "__main__":
    test_eig_property()
    print("All tests passed!")
```

<details>

<summary>
**Failing input**: `array([[0.00000000e+000, 1.85401547e-286, 0.00000000e+000], [1.00000000e+000, 1.00000000e+000, 0.00000000e+000], [0.00000000e+000, 0.00000000e+000, 0.00000000e+000]])`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 28, in <module>
    test_eig_property()
    ~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 7, in test_eig_property
    @settings(max_examples=300)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 24, in test_eig_property
    assert np.allclose(lhs, rhs, rtol=1e-4, atol=1e-7), f"A @ v != lambda * v for eigenpair {i}"
           ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: A @ v != lambda * v for eigenpair 0
Falsifying example: test_eig_property(
    a=array([[0.00000000e+000, 1.85401547e-286, 0.00000000e+000],
           [1.00000000e+000, 1.00000000e+000, 0.00000000e+000],
           [0.00000000e+000, 0.00000000e+000, 0.00000000e+000]]),
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
import numpy.linalg as la

# The problematic matrix with extreme value
a = np.array([[0.00000000e+00, 1.17549435e-38, 0.00000000e+00],
              [1.00000000e+00, 1.00000000e+00, 0.00000000e+00],
              [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]])

print("Matrix A:")
print(a)
print()

# Compute eigenvalues and eigenvectors
result = la.eig(a)
eigenvalues = result.eigenvalues
eigenvectors = result.eigenvectors

print("Eigenvalues:")
print(eigenvalues)
print()

print("Eigenvectors (columns):")
print(eigenvectors)
print()

# Check each eigenpair
for i in range(len(eigenvalues)):
    lambda_i = eigenvalues[i]
    v_i = eigenvectors[:, i]

    print(f"Eigenpair {i}:")
    print(f"  Eigenvalue λ = {lambda_i}")
    print(f"  Eigenvector v = {v_i}")

    # Check if A @ v = λ * v
    lhs = a @ v_i
    rhs = lambda_i * v_i

    print(f"  A @ v = {lhs}")
    print(f"  λ * v = {rhs}")

    # Check if they are equal within reasonable tolerance
    is_equal = np.allclose(lhs, rhs, rtol=1e-4, atol=1e-7)
    print(f"  A @ v ≈ λ * v? {is_equal}")

    if not is_equal:
        error = np.linalg.norm(lhs - rhs)
        print(f"  Error magnitude: {error}")
    print()
```

<details>

<summary>
First eigenpair violates eigenvalue equation with error magnitude 1.0
</summary>
```
Matrix A:
[[0.00000000e+00 1.17549435e-38 0.00000000e+00]
 [1.00000000e+00 1.00000000e+00 0.00000000e+00]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00]]

Eigenvalues:
[0. 1. 0.]

Eigenvectors (columns):
[[1.00000000e+00 1.17549435e-38 0.00000000e+00]
 [0.00000000e+00 1.00000000e+00 0.00000000e+00]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]

Eigenpair 0:
  Eigenvalue λ = 0.0
  Eigenvector v = [1. 0. 0.]
  A @ v = [0. 1. 0.]
  λ * v = [0. 0. 0.]
  A @ v ≈ λ * v? False
  Error magnitude: 1.0

Eigenpair 1:
  Eigenvalue λ = 1.0
  Eigenvector v = [1.17549435e-38 1.00000000e+00 0.00000000e+00]
  A @ v = [1.17549435e-38 1.00000000e+00 0.00000000e+00]
  λ * v = [1.17549435e-38 1.00000000e+00 0.00000000e+00]
  A @ v ≈ λ * v? True

Eigenpair 2:
  Eigenvalue λ = 0.0
  Eigenvector v = [0. 0. 1.]
  A @ v = [0. 0. 0.]
  λ * v = [0. 0. 0.]
  A @ v ≈ λ * v? True
```
</details>

## Why This Is A Bug

The eigenvalue equation A @ v = λ * v is the fundamental mathematical definition of eigenvalues and eigenvectors. For the first returned eigenpair (λ=0, v=[1,0,0]), the function violates this equation:

- A @ v = [0, 1, 0]
- λ * v = [0, 0, 0]
- Error magnitude: 1.0

This is not a minor numerical inaccuracy but a complete mathematical failure. The eigenvector [1, 0, 0] is simply incorrect for eigenvalue 0. The correct eigenvector should satisfy A @ v = 0 * v = [0, 0, 0].

The numpy.linalg.eig documentation explicitly states that it returns eigenvectors such that `a @ eigenvectors[:,i] = eigenvalues[i] * eigenvectors[:,i]`. This bug violates this documented contract without raising any warnings or errors.

## Relevant Context

1. The bug occurs specifically when matrices contain extremely small values near machine epsilon (e.g., 1.175e-38, which is the smallest normal float32 value) mixed with normal-scale values.

2. The matrix has an infinite condition number and is singular (determinant = 0), making it numerically challenging. However, the error magnitude of 1.0 far exceeds any reasonable numerical tolerance.

3. As of NumPy 2.3.0, `numpy.linalg.eig` returns a namedtuple `EigResult` with `eigenvalues` and `eigenvectors` attributes, not a simple tuple as in older versions.

4. The other two eigenpairs returned by the function are correct and satisfy the eigenvalue equation within numerical tolerance.

5. The issue appears to be in the underlying LAPACK `_geev` routines that numpy uses for eigenvalue decomposition, specifically in handling matrices with extreme value ranges.

## Proposed Fix

This bug requires improvements to numerical stability in the eigenvalue decomposition algorithm. A high-level approach would involve:

1. Pre-conditioning the matrix to improve numerical stability when extreme value ranges are detected
2. Adding post-computation verification that eigenpairs satisfy A @ v ≈ λ * v within tolerance
3. Implementing iterative refinement for eigenpairs that fail verification
4. Adding warnings when the condition number exceeds safe thresholds

Since this involves changes to core numerical algorithms potentially in LAPACK itself, a workaround at the NumPy level could add validation and warnings:

```diff
# Pseudo-code for numpy.linalg.eig wrapper
def eig(a):
    eigenvalues, eigenvectors = _lapack_geev(a)

+   # Check for numerical stability issues
+   for i in range(len(eigenvalues)):
+       v = eigenvectors[:, i]
+       error = np.linalg.norm(a @ v - eigenvalues[i] * v)
+       if error > max(1e-7, 1e-4 * np.linalg.norm(a)):
+           import warnings
+           warnings.warn(
+               f"Eigenpair {i} failed verification with error {error}. "
+               "Results may be unreliable due to numerical instability.",
+               RuntimeWarning
+           )

    return EigResult(eigenvalues, eigenvectors)
```