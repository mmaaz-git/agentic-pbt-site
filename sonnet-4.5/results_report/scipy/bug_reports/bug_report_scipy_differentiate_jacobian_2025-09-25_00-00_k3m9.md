# Bug Report: scipy.differentiate.jacobian Returns Transposed Jacobian Matrix

**Target**: `scipy.differentiate.jacobian`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `jacobian` function returns the transpose of the Jacobian matrix instead of the Jacobian matrix itself. For a linear function f(x) = Ax, the Jacobian should be A, but the function returns A.T.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st
from scipy.differentiate import jacobian


@given(st.integers(min_value=2, max_value=5))
def test_jacobian_of_linear_function(m):
    rng = np.random.default_rng()
    A = rng.standard_normal((m, m))

    def f(xi):
        return A @ xi

    x = rng.standard_normal(m)
    res = jacobian(f, x)

    if res.success.all():
        assert np.allclose(res.df, A, rtol=1e-4), \
            f"Jacobian should equal A, but got A.T"
```

**Failing input**: Any non-symmetric matrix

## Reproducing the Bug

```python
import numpy as np
from scipy.differentiate import jacobian

A = np.array([[1.0, 2.0, 3.0],
              [0.0, 4.0, 5.0],
              [0.0, 0.0, 6.0]])

def f(xi):
    return A @ xi

x = np.zeros(3)

res = jacobian(f, x)

print("For f(x) = Ax, the Jacobian should be A:")
print(A)
print("\nBut scipy returns A.T:")
print(res.df)
print("\nVerify: res.df == A.T:", np.allclose(res.df, A.T))
```

## Why This Is A Bug

The Jacobian matrix J of a function f: ℝ^m → ℝ^n is defined as J[i,j] = ∂f_i/∂x_j.

For a linear function f(x) = Ax where A is an m×m matrix:
- f_i = Σ_k A[i,k] * x[k]
- ∂f_i/∂x_j = A[i,j]

Therefore, the Jacobian J should equal A.

However, `scipy.differentiate.jacobian` returns A.T (the transpose), which means it's computing J[j,i] = ∂f_i/∂x_j instead of J[i,j] = ∂f_i/∂x_j.

This violates the standard mathematical definition of the Jacobian matrix and contradicts the function's own documentation, which states that for a function mapping ℝ^m → ℝ^n, the Jacobian should have shape (n, m) with element [i,j] representing ∂f_i/∂x_j.

## Fix

The bug is likely in how the derivative results are assembled into the Jacobian matrix. Looking at the code structure from the traceback, `jacobian` calls `derivative` in a wrapped form. The issue appears to be in how the results from varying each input component are assembled.

The fix would involve transposing the result before returning it, or fixing the order in which derivatives are computed and assembled. Without access to the full source code, the specific line would need to be identified, but the fix should be approximately:

```diff
-    return result.df
+    return result.df.T
```

Or alternatively, the indexing order when building the Jacobian needs to be swapped to match the standard mathematical convention.