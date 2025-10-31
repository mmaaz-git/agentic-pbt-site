# Bug Report: scipy.differentiate.jacobian Returns Transposed Jacobian Matrix

**Target**: `scipy.differentiate.jacobian`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `jacobian` function returns the transpose of the mathematically correct Jacobian matrix, violating both its documentation and the standard mathematical definition.

## Property-Based Test

```python
import numpy as np
from scipy.differentiate import jacobian
from hypothesis import given, strategies as st, settings

@given(size=st.integers(min_value=2, max_value=5))
@settings(max_examples=50)
def test_jacobian_linear_function(size):
    rng = np.random.RandomState(42)
    A = rng.randn(size, size)

    def f(x):
        return A @ x

    x = rng.randn(size)
    res = jacobian(f, x)

    if np.all(res.success):
        assert np.allclose(res.df, A, rtol=1e-4, atol=1e-6), \
            f"Jacobian mismatch: got {res.df}, expected {A}"

if __name__ == "__main__":
    test_jacobian_linear_function()
```

<details>

<summary>
**Failing input**: `size=2`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/59/hypo.py", line 22, in <module>
    test_jacobian_linear_function()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/59/hypo.py", line 6, in test_jacobian_linear_function
    @settings(max_examples=50)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/59/hypo.py", line 18, in test_jacobian_linear_function
    assert np.allclose(res.df, A, rtol=1e-4, atol=1e-6), \
           ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Jacobian mismatch: got [[ 0.49671415  0.64768854]
 [-0.1382643   1.52302986]], expected [[ 0.49671415 -0.1382643 ]
 [ 0.64768854  1.52302986]]
Falsifying example: test_jacobian_linear_function(
    size=2,
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
from scipy.differentiate import jacobian

# Simple test case: linear function f(x) = Ax
A = np.array([[1.0, 2.0],
              [3.0, 5.0]])

def f(x):
    return A @ x

x = np.array([1.0, 2.0])
result = jacobian(f, x)

print("Matrix A:")
print(A)
print("\nComputed Jacobian:")
print(result.df)
print("\nExpected (A):")
print(A)
print("\nActual result (A.T):")
print(A.T)
print("\nMatch with A.T:", np.allclose(result.df, A.T))
print("\nMatch with A:", np.allclose(result.df, A))

# Verify the mathematical definition
print("\n--- Mathematical Verification ---")
print("For linear function f(x) = Ax:")
print("Jacobian J[i,j] should equal A[i,j] = ∂f_i/∂x_j")
print("\nChecking elements:")
for i in range(2):
    for j in range(2):
        print(f"J[{i},{j}] = {result.df[i,j]:.1f}, A[{i},{j}] = {A[i,j]:.1f}, A.T[{i},{j}] = {A.T[i,j]:.1f}")
```

<details>

<summary>
Output showing transposed result
</summary>
```
Matrix A:
[[1. 2.]
 [3. 5.]]

Computed Jacobian:
[[1. 3.]
 [2. 5.]]

Expected (A):
[[1. 2.]
 [3. 5.]]

Actual result (A.T):
[[1. 3.]
 [2. 5.]]

Match with A.T: True

Match with A: False

--- Mathematical Verification ---
For linear function f(x) = Ax:
Jacobian J[i,j] should equal A[i,j] = ∂f_i/∂x_j

Checking elements:
J[0,0] = 1.0, A[0,0] = 1.0, A.T[0,0] = 1.0
J[0,1] = 3.0, A[0,1] = 2.0, A.T[0,1] = 3.0
J[1,0] = 2.0, A[1,0] = 3.0, A.T[1,0] = 2.0
J[1,1] = 5.0, A[1,1] = 5.0, A.T[1,1] = 5.0
```
</details>

## Why This Is A Bug

The Jacobian matrix is mathematically defined such that element J[i,j] = ∂f_i/∂x_j, where f_i is the i-th component of the output and x_j is the j-th component of the input.

For the linear function f(x) = Ax, the Jacobian should be exactly the matrix A. However, the function returns A.T instead, meaning that J[i,j] = ∂f_j/∂x_i (indices are swapped).

The documentation for `scipy.differentiate.jacobian` explicitly states that for a function f: R^m → R^n, the Jacobian should be an n×m matrix where element [i,j] represents ∂f_i/∂x_j. The current implementation violates this documented behavior.

This bug affects all uses of the jacobian function and will cause incorrect results in:
- Gradient-based optimization algorithms
- Newton's method and quasi-Newton methods
- Sensitivity analysis
- Machine learning applications using automatic differentiation
- Any scientific computing application relying on accurate Jacobian computation

## Relevant Context

The scipy.differentiate module was introduced in SciPy 1.14.0 as a new module for numerical differentiation. The jacobian function is a core feature of this module and is expected to follow the standard mathematical definition.

The bug occurs because the internal implementation constructs a wrapped function that is passed to the `derivative` function. The way the wrapped function is constructed and how `derivative` computes element-wise derivatives leads to the transposed result.

Related documentation:
- SciPy differentiate module: https://docs.scipy.org/doc/scipy/reference/differentiate.html
- Mathematical definition of Jacobian: https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant

## Proposed Fix

The fix is to transpose the result before returning it from the `jacobian` function. This can be done using numpy's `moveaxis` to handle both 2D and higher-dimensional cases correctly:

```diff
--- a/scipy/differentiate/_differentiate.py
+++ b/scipy/differentiate/_differentiate.py
@@ -937,6 +937,7 @@ def jacobian(f, x, *, tolerances=None, maxiter=10, order=8, initial_step=0.5,
                      step_factor=step_factor, preserve_shape=True,
                      step_direction=step_direction)

+    res.df = xp.moveaxis(res.df, 0, 1)  # Transpose to get correct index order
     del res.x  # the user knows `x`, and the way it gets broadcasted is meaningless here
     return res
```