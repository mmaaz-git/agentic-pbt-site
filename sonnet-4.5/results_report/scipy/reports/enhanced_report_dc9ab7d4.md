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


if __name__ == "__main__":
    test_jacobian_of_linear_function()
```

<details>

<summary>
**Failing input**: `m=2`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 23, in <module>
    test_jacobian_of_linear_function()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 7, in test_jacobian_of_linear_function
    def test_jacobian_of_linear_function(m):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 18, in test_jacobian_of_linear_function
    assert np.allclose(res.df, A, rtol=1e-4), \
           ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Jacobian should equal A, but got A.T
Falsifying example: test_jacobian_of_linear_function(
    m=2,
)
```
</details>

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
print("\nBut scipy returns:")
print(res.df)
print("\nVerify: res.df == A.T:", np.allclose(res.df, A.T))
print("Verify: res.df == A:", np.allclose(res.df, A))
```

<details>

<summary>
Output demonstrating transposed Jacobian
</summary>
```
For f(x) = Ax, the Jacobian should be A:
[[1. 2. 3.]
 [0. 4. 5.]
 [0. 0. 6.]]

But scipy returns:
[[1. 0. 0.]
 [2. 4. 0.]
 [3. 5. 6.]]

Verify: res.df == A.T: True
Verify: res.df == A: False
```
</details>

## Why This Is A Bug

The Jacobian matrix J of a function f: ℝ^m → ℝ^n is mathematically defined as J[i,j] = ∂f_i/∂x_j, where:
- f_i is the i-th component of the output
- x_j is the j-th component of the input

For a linear function f(x) = Ax where A is an n×m matrix:
- The i-th output is f_i(x) = Σ_k A[i,k] * x[k]
- The partial derivative ∂f_i/∂x_j = A[i,j]
- Therefore, the Jacobian J should equal A

However, `scipy.differentiate.jacobian` returns a matrix where element [i,j] contains A[j,i], which is the transpose of the correct Jacobian. This violates:

1. **Standard Mathematical Definition**: The universally accepted definition in mathematics states J[i,j] = ∂f_i/∂x_j
2. **SciPy's Own Documentation**: Lines 818-820 in the source code state that for f: ℝ^m → ℝ^n, the result `df` should be an (n, m) array representing the Jacobian
3. **Expected Behavior**: Every mathematical software package (MATLAB, Julia, Mathematica) follows the standard convention

The bug occurs because the implementation incorrectly assembles the derivative results when varying each input component.

## Relevant Context

The bug is located in the `jacobian` function implementation at `/home/npc/pbt/agentic-pbt/envs/scipy_env/lib/python3.13/site-packages/scipy/differentiate/_differentiate.py:912-937`. The issue stems from how the `wrapped` function constructs the variation pattern for computing derivatives.

The function creates a matrix `xph` where it varies each component of the input one at a time (line 928). However, the indexing pattern used causes the derivatives to be assembled in transposed order. When `derivative` is called on this wrapped function, it computes the correct numerical values but places them in the wrong positions of the result matrix.

Documentation reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.differentiate.jacobian.html

## Proposed Fix

The bug can be fixed by transposing the result before returning it. Here's the patch:

```diff
--- a/scipy/differentiate/_differentiate.py
+++ b/scipy/differentiate/_differentiate.py
@@ -933,6 +933,9 @@ def jacobian(f, x, *, tolerances=None, maxiter=10, order=8, initial_step=0.5,
                      step_factor=step_factor, preserve_shape=True,
                      step_direction=step_direction)

+    # Fix: The derivative computation returns the transpose of the Jacobian
+    res.df = xp.permute_dims(res.df, list(range(res.df.ndim - 1, -1, -1)))
+
     del res.x  # the user knows `x`, and the way it gets broadcasted is meaningless here
     return res
```

Alternatively, the indexing in the `wrapped` function could be modified to produce the correct orientation from the start, but that would require more extensive changes to the derivative computation logic.