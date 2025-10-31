# Bug Report: scipy.differentiate.derivative Crash with step_factor Near 1.0

**Target**: `scipy.differentiate.derivative`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `derivative` function crashes with `numpy.linalg.LinAlgError: Singular matrix` when `step_factor` is at or very close to 1.0, despite no documented restriction on this value.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
from scipy.differentiate import derivative

@given(
    step_factor=st.floats(min_value=1.0001, max_value=1.01, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100)
def test_step_factor_close_to_one(step_factor):
    def f(x_val):
        return x_val ** 2

    x = 1.5
    res = derivative(f, x, step_factor=step_factor, maxiter=3)

test_step_factor_close_to_one()
```

<details>

<summary>
**Failing input**: `step_factor=1.0021992196575686`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/9/hypo.py", line 16, in <module>
    test_step_factor_close_to_one()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/9/hypo.py", line 6, in test_step_factor_close_to_one
    step_factor=st.floats(min_value=1.0001, max_value=1.01, allow_nan=False, allow_infinity=False)
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/9/hypo.py", line 14, in test_step_factor_close_to_one
    res = derivative(f, x, step_factor=step_factor, maxiter=3)
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/differentiate/_differentiate.py", line 586, in derivative
    return eim._loop(work, callback, shape, maxiter, func, args, dtype,
           ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                     pre_func_eval, post_func_eval, check_termination,
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                     post_termination_check, customize_result, res_work_pairs,
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                     xp, preserve_shape)
                     ^^^^^^^^^^^^^^^^^^^
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/_lib/_elementwise_iterative_method.py", line 250, in _loop
    post_func_eval(x, f, work)
    ~~~~~~~~~~~~~~^^^^^^^^^^^^
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/differentiate/_differentiate.py", line 536, in post_func_eval
    wc, wo = _derivative_weights(work, n, xp)
             ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/differentiate/_differentiate.py", line 704, in _derivative_weights
    weights = np.linalg.solve(A, b)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/linalg/_linalg.py", line 471, in solve
    r = gufunc(a, b, signature=signature)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/linalg/_linalg.py", line 163, in _raise_linalgerror_singular
    raise LinAlgError("Singular matrix")
numpy.linalg.LinAlgError: Singular matrix
Falsifying example: test_step_factor_close_to_one(
    step_factor=1.0021992196575686,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/linalg/_linalg.py:163
```
</details>

## Reproducing the Bug

```python
import numpy as np
from scipy.differentiate import derivative

def f(x_val):
    return x_val ** 2

x = 1.0

for step_factor in [1.0, 1.0003, 1.001, 1.01, 1.1]:
    try:
        res = derivative(f, x, step_factor=step_factor, maxiter=2)
        print(f"step_factor={step_factor:.4f}: SUCCESS, df={res.df:.6f}")
    except np.linalg.LinAlgError as e:
        print(f"step_factor={step_factor:.4f}: CRASH - {e}")
```

<details>

<summary>
Output showing crash with step_factor=1.0
</summary>
```
step_factor=1.0000: CRASH - Singular matrix
step_factor=1.0003: SUCCESS, df=2.000003
step_factor=1.0010: SUCCESS, df=2.000000
step_factor=1.0100: SUCCESS, df=2.000000
step_factor=1.1000: SUCCESS, df=2.000000
```
</details>

## Why This Is A Bug

This violates expected behavior in several ways:

1. **Invalid input should be rejected gracefully**: The input validation in `_derivative_iv` (lines 27-33 of `/home/npc/.local/lib/python3.13/site-packages/scipy/differentiate/_differentiate.py`) checks that `step_factor >= 0` but doesn't reject values at or near 1.0. Instead, the function crashes deep in the implementation with an obscure linear algebra error.

2. **Documentation doesn't mention this restriction**: The docstring describes `step_factor` as "The factor by which the step size is *reduced* in each iteration" (line 110) with no warning that values at or near 1.0 are invalid. The documentation specifically states that values less than 1.0 are allowed for cases where larger steps are desired.

3. **Root cause - numerical instability**: In `_derivative_weights` (lines 674-704), the finite difference weights are computed by solving a linear system. The code constructs step sizes using:
   - For central differences: `h = s / fac**p` where `fac` is the step_factor and `p` varies
   - When `fac = 1.0`, all powers `fac**p` equal 1.0
   - This makes all step sizes identical (either 1 or -1)
   - The Vandermonde matrix `A = np.vander(h, increasing=True).T` becomes singular
   - `np.linalg.solve(A, b)` fails with a singular matrix error

4. **User impact**: Users might reasonably set `step_factor=1.0` thinking it means "no step reduction" between iterations, or this value could arise from configuration calculations. They would receive a cryptic linear algebra error instead of a clear validation message explaining the issue.

## Relevant Context

The finite difference method relies on evaluating the function at different step sizes to estimate derivatives. When `step_factor=1.0`, all iterations use identical step sizes, making it impossible to construct the finite difference formula. The algorithm requires distinct step sizes to form a well-conditioned linear system for computing the weights.

The code in `_derivative_weights` specifically at line 678 creates the step pattern `h = s / fac ** p`. When `fac=1.0`, this degenerates to `h = s` for all power values, creating duplicate evaluation points that lead to a singular Vandermonde matrix.

## Proposed Fix

Add validation in `_derivative_iv` to reject `step_factor` values too close to 1.0:

```diff
--- a/scipy/differentiate/_differentiate.py
+++ b/scipy/differentiate/_differentiate.py
@@ -30,6 +30,9 @@ def _derivative_iv(f, x, args, tolerances, maxiter, order, initial_step,
                        step_factor])
     if (not np.issubdtype(tols.dtype, np.number) or np.any(tols < 0)
             or np.any(np.isnan(tols)) or tols.shape != (3,)):
         raise ValueError(message)
+    if abs(step_factor - 1.0) < 0.01:
+        raise ValueError('`step_factor` must not be too close to 1.0 (must satisfy |step_factor - 1.0| >= 0.01). '
+                        'A step_factor of 1.0 means no step size reduction between iterations, '
+                        'which prevents the algorithm from computing finite difference weights.')
     step_factor = float(tols[2])
```