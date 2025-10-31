# Bug Report: scipy.differentiate.jacobian Returns Incorrect Results with @ and np.matmul Operators

**Target**: `scipy.differentiate.jacobian`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `jacobian` function returns mathematically incorrect Jacobian matrices when the input function uses the `@` operator or `np.matmul` for matrix multiplication, due to incompatible array broadcasting behavior with higher-dimensional arrays.

## Property-Based Test

```python
import numpy as np
from scipy.differentiate import jacobian
from hypothesis import given, strategies as st, settings, assume

@given(
    x=st.lists(
        st.floats(min_value=-5, max_value=5, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=4
    )
)
@settings(max_examples=50)
def test_jacobian_linear_function(x):
    x_arr = np.array(x)
    n = len(x_arr)

    A = np.random.RandomState(42).randn(n + 1, n)

    def f(xi):
        return A @ xi

    res = jacobian(f, x_arr)
    assume(np.all(res.success))

    expected = A

    assert np.allclose(res.df, expected, rtol=1e-5, atol=1e-7)

if __name__ == "__main__":
    test_jacobian_linear_function()
```

<details>

<summary>
**Failing input**: `x=[0.0, 0.0]` (or any input)
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/40/hypo.py", line 30, in <module>
    test_jacobian_linear_function()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/40/hypo.py", line 6, in test_jacobian_linear_function
    x=st.lists(

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/40/hypo.py", line 27, in test_jacobian_linear_function
    assert np.allclose(res.df, expected, rtol=1e-5, atol=1e-7)
           ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_jacobian_linear_function(
    x=[0.0, 0.0],  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
from scipy.differentiate import jacobian

x = np.array([1.0, 2.0])

A = np.array([[1.0, 2.0],
              [3.0, 4.0],
              [5.0, 6.0]])

def f(xi):
    return A @ xi

result = jacobian(f, x)

print("Expected Jacobian:")
print(A)

print("\nActual Jacobian:")
print(result.df)

epsilon = 1e-8
manual = []
for i in range(len(x)):
    e = np.zeros_like(x)
    e[i] = epsilon
    df = (f(x + e) - f(x - e)) / (2 * epsilon)
    manual.append(df)
manual = np.array(manual).T

print("\nManual calculation:")
print(manual)

print("\nComparison:")
print(f"Are they equal? {np.allclose(result.df, A, rtol=1e-5, atol=1e-7)}")
print(f"Max absolute difference: {np.max(np.abs(result.df - A))}")
```

<details>

<summary>
Bug manifests as scrambled Jacobian matrix elements
</summary>
```
Expected Jacobian:
[[1. 2.]
 [3. 4.]
 [5. 6.]]

Actual Jacobian:
[[1. 3.]
 [5. 2.]
 [4. 6.]]

Manual calculation:
[[0.99999999 1.99999999]
 [2.99999998 3.99999998]
 [5.00000006 5.99999979]]

Comparison:
Are they equal? False
Max absolute difference: 2.0000000000000133
```
</details>

## Why This Is A Bug

For a linear function `f(x) = A @ x`, the Jacobian is mathematically equal to the matrix `A`. The `jacobian` function should compute J[i,j] = ∂f_i/∂x_j, which for a linear transformation is exactly A[i,j].

The bug occurs because:

1. The `jacobian` function internally vectorizes the computation by calling `f` with arrays of shape `(m, m, ...)` where `m` is the dimension of the input
2. When using `@` or `np.matmul`, NumPy's broadcasting rules for generalized matrix multiplication produce unexpected results:
   - For `A` of shape `(3, 2)` and `xi` of shape `(2, 2, 8)`, `A @ xi` produces shape `(2, 3, 8)` instead of the expected `(3, 2, 8)`
   - This is because `@` treats the last two dimensions as matrix dimensions for generalized matrix multiplication
3. The scrambled output shape leads to incorrect element ordering in the final Jacobian

The bug **only** affects functions using `@` or `np.matmul`. Functions using `np.dot` work correctly because `np.dot` handles broadcasting differently, producing the expected `(3, 2, 8)` shape.

## Relevant Context

This is a serious bug because:
- The `@` operator is the modern, Pythonic way to do matrix multiplication (PEP 465)
- `np.matmul` is the recommended function equivalent of `@`
- The bug produces silently incorrect results with no warnings
- Incorrect Jacobians will propagate errors through optimization, root-finding, and other numerical algorithms
- Users have no indication they need to avoid these standard operators

Documentation references:
- NumPy matmul documentation: https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
- The wrapped function code is in `/home/npc/.local/lib/python3.13/site-packages/scipy/differentiate/_differentiate.py:920-929`

## Proposed Fix

The issue is in the `wrapped` function within `jacobian` (lines 920-929 of `_differentiate.py`). The function needs to handle the broadcasting behavior of `@`/`np.matmul` differently. A workaround is to document that users should use `np.dot` instead, but the proper fix would be to make the wrapper compatible with all standard NumPy matrix operations.

Since the fix is complex and would require careful handling of different array broadcasting rules, I recommend:
1. **Immediate**: Add a warning in the documentation about this limitation with `@` and `np.matmul`
2. **Long-term**: Redesign the internal vectorization to be compatible with generalized matrix multiplication broadcasting rules