# Bug Report: scipy.differentiate.jacobian Returns Transposed Jacobian

**Target**: `scipy.differentiate.jacobian`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `jacobian` function returns the transpose of the Jacobian matrix. For a linear function f(x) = Ax, the Jacobian should be A, but the function returns A.T instead.

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
```

**Failing input**: `size=2` (or any size)

## Reproducing the Bug

```python
import numpy as np
from scipy.differentiate import jacobian

A = np.array([[1.0, 2.0],
              [3.0, 5.0]])

def f(x):
    return A @ x

x = np.array([1.0, 2.0])
result = jacobian(f, x)

print(f"Matrix A:\n{A}")
print(f"\nComputed Jacobian:\n{result.df}")
print(f"\nExpected (A):\n{A}")
print(f"\nActual result (A.T):\n{A.T}")
print(f"\nMatch with A.T: {np.allclose(result.df, A.T)}")
```

Output:
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
```

## Why This Is A Bug

For a function f: R^m → R^n, the Jacobian J is defined as an n×m matrix where J[i,j] = ∂f_i/∂x_j.

For the linear function f(x) = Ax:
- f_i(x) = Σ_k A[i,k] · x[k]
- ∂f_i/∂x_j = A[i,j]
- Therefore J = A

The documentation for `jacobian` confirms this definition. However, the function returns A.T instead of A, which means J[i,j] = ∂f_j/∂x_i (indices swapped).

This is a critical bug that affects all uses of the `jacobian` function, as it returns incorrect derivatives that will lead to wrong results in optimization, scientific computing, and machine learning applications.

## Root Cause

The bug occurs in the `jacobian` function implementation in `scipy/differentiate/_differentiate.py`. The function constructs a `wrapped` function and calls `derivative` on it.

The issue is that `derivative` computes element-wise derivatives, so `res.df[i, j]` contains `d(wrapped(x)[i, j])/dx[i]`, not `d(wrapped(x)[i, j])/dx[j]`.

Due to how `wrapped` constructs the input array `xph`:
- `xph[i, i] = x[i]` (diagonal elements are the perturbed inputs)
- `f(xph)[output_idx, col_idx]` represents the output when column `col_idx` of `xph` is fed to `f`
- Column `i` of `xph` has the `i`-th input component perturbed

Therefore:
- `wrapped(x)[i, j]` = `f(xph)[i, j]` = j-th output component when i-th input is perturbed
- `d(wrapped(x)[i, j])/dx[i]` = `∂f_j/∂x_i`
- So `res.df[i, j]` = `∂f_j/∂x_i` = `J[j, i]` (transposed!)

## Fix

The fix is to transpose the result before returning it from the `jacobian` function:

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

Alternatively, the axes can be swapped when constructing `xph` or when indexing the result, but transposing the final result is the simplest fix. For vectorized cases where the Jacobian shape is `(n, m, k)` for k evaluation points, we need `moveaxis(res.df, 0, 1)` to swap the first two axes while preserving the batch dimension.