# Bug Report: scipy.differentiate.jacobian Returns Transposed Jacobian for Matrix Multiplication

**Target**: `scipy.differentiate.jacobian`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When computing the Jacobian of a linear transformation `f(x) = A @ x` using `scipy.differentiate.jacobian`, the function returns the transpose of the correct Jacobian matrix. This is a fundamental mathematical error that affects any function using numpy's `@` operator with multi-dimensional arrays.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings
from scipy.differentiate import jacobian


@given(
    n=st.integers(min_value=2, max_value=5),
    m=st.integers(min_value=2, max_value=5)
)
@settings(max_examples=50)
def test_jacobian_linear_transformation_property(n, m):
    A = np.random.randn(m, n).astype(np.float64)

    def f(x):
        return A @ x

    x = np.random.randn(n).astype(np.float64)

    result = jacobian(f, x)
    J = result.df

    np.testing.assert_allclose(
        J, A, rtol=1e-4, atol=1e-6,
        err_msg="Jacobian of f(x)=Ax should equal A"
    )
```

**Failing input**: `n=2, m=2` (fails for all non-diagonal matrices)

## Reproducing the Bug

```python
import numpy as np
from scipy.differentiate import jacobian

A = np.array([[1.0, 2.0], [3.0, 4.0]])

def f(x):
    return A @ x

x = np.array([1.0, 1.0])
result = jacobian(f, x)

print("Matrix A:")
print(A)
print("\nComputed Jacobian:")
print(result.df)
print("\nExpected: A")
print("Got: A.T")
```

Output:
```
Matrix A:
[[1. 2.]
 [3. 4.]]

Computed Jacobian:
[[1. 3.]
 [2. 4.]]

Expected: A
Got: A.T
```

## Why This Is A Bug

For a linear transformation f(x) = Ax where A is an m×n matrix:
- f_i(x) = Σ_j A[i,j] * x[j]
- The Jacobian is ∂f_i/∂x_j = A[i,j]
- Therefore, the Jacobian matrix J should equal A

However, `scipy.differentiate.jacobian` returns A.T (the transpose) when the function is implemented using numpy's `@` operator.

This bug does NOT occur when the function is written explicitly. For example, this works correctly:

```python
def f_explicit(x):
    x1, x2 = x
    return np.array([x1 + 2*x2, 3*x1 + 4*x2])
```

The issue appears to be related to how numpy's `@` operator broadcasts with multi-dimensional arrays. When `jacobian` internally calls the function with arrays of shape `(m, k, ...)` for batched finite-difference computations, the `@` operator treats the first dimension as a batch dimension rather than the input components dimension, causing the axes to be swapped.

## Fix

The root cause is that `scipy.differentiate.jacobian` expects functions to handle multi-dimensional inputs with the convention that axis 0 represents input components. However, when using `A @ x` with `x` of shape `(m, k, ...)`, numpy's matmul treats axis 0 of `x` as a batch dimension.

**Workaround for users**: Explicitly unpack the input and construct the output:

```python
def f_correct(x):
    x1, x2 = x
    return np.array([
        A[0,0]*x1 + A[0,1]*x2,
        A[1,0]*x1 + A[1,1]*x2
    ])
```

**Potential fix for scipy**: The documentation should clarify this behavior, or `jacobian` should transpose the result when detecting matrix multiplication patterns. Alternatively, the internal implementation should reshape arrays to avoid this broadcasting issue.

A more fundamental fix would be to modify how `jacobian` batches its inputs to be compatible with standard numpy broadcasting conventions for `@`.