# Bug Report: scipy.differentiate.derivative Crashes with Singular Matrix Error When step_factor=1.0

**Target**: `scipy.differentiate.derivative`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `scipy.differentiate.derivative` function crashes with a `LinAlgError: Singular matrix` when `step_factor=1.0` is passed, even though this value passes all input validation checks. The function should either handle this case gracefully or provide a clear error message about the constraint.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, assume, example
from scipy.differentiate import derivative
import numpy as np
import math

@settings(max_examples=50)
@given(
    initial_step=st.floats(min_value=0.01, max_value=10, allow_nan=False, allow_infinity=False),
    step_factor=st.floats(min_value=0.1, max_value=5, allow_nan=False, allow_infinity=False)
)
@example(initial_step=0.5, step_factor=1.0)  # This should trigger the bug
def test_step_parameters_produce_valid_results(initial_step, step_factor):
    assume(step_factor > 0.05)
    x = 1.5
    res = derivative(np.exp, x, initial_step=initial_step, step_factor=step_factor)
    if res.success:
        expected = np.exp(x)
        assert math.isclose(res.df, expected, rel_tol=1e-5)

# Run the test
if __name__ == "__main__":
    test_step_parameters_produce_valid_results()
```

<details>

<summary>
**Failing input**: `initial_step=0.5, step_factor=1.0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 22, in <module>
    test_step_parameters_produce_valid_results()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 7, in test_step_parameters_produce_valid_results
    @given(

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
    _raise_to_user(errors, state.settings, [], " in explicit examples")
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 15, in test_step_parameters_produce_valid_results
    res = derivative(np.exp, x, initial_step=initial_step, step_factor=step_factor)
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
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/differentiate/_differentiate.py", line 682, in _derivative_weights
    weights = np.linalg.solve(A, b)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/linalg/_linalg.py", line 471, in solve
    r = gufunc(a, b, signature=signature)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/linalg/_linalg.py", line 163, in _raise_linalgerror_singular
    raise LinAlgError("Singular matrix")
numpy.linalg.LinAlgError: Singular matrix
Falsifying explicit example: test_step_parameters_produce_valid_results(
    initial_step=0.5,
    step_factor=1.0,
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
from scipy.differentiate import derivative

# Reproduce the bug with step_factor=1.0
res = derivative(np.exp, 1.5, step_factor=1.0)
print(res)
```

<details>

<summary>
LinAlgError: Singular matrix
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/52/repo.py", line 5, in <module>
    res = derivative(np.exp, 1.5, step_factor=1.0)
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
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/differentiate/_differentiate.py", line 682, in _derivative_weights
    weights = np.linalg.solve(A, b)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/linalg/_linalg.py", line 471, in solve
    r = gufunc(a, b, signature=signature)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/linalg/_linalg.py", line 163, in _raise_linalgerror_singular
    raise LinAlgError("Singular matrix")
numpy.linalg.LinAlgError: Singular matrix
```
</details>

## Why This Is A Bug

1. **Input validation accepts `step_factor=1.0`**: The validation code in `_derivative_iv` (lines 31-33) only checks that `step_factor >= 0` and is not NaN. There is no check for `step_factor == 1.0`, so this value passes validation even though it causes a crash later.

2. **Documentation doesn't indicate the constraint**: The documentation for `step_factor` states it's "The factor by which the step size is *reduced* in each iteration" with a default of 2.0. It mentions that `step_factor < 1` may be useful but doesn't warn that `step_factor = 1.0` is invalid.

3. **Mathematical cause of the crash**: When `step_factor=1.0`, the step size remains constant across iterations (h/1 = h). In the `_derivative_weights` function (line 678), the formula `h = s / fac ** p` produces identical values when `fac=1.0` because `1.0 ** p = 1.0` for all powers `p`. This creates a Vandermonde matrix with repeated rows, which is singular and cannot be inverted by `np.linalg.solve` (line 682).

4. **Poor error message**: Users receive a cryptic `numpy.linalg.LinAlgError: Singular matrix` from deep within NumPy's linear algebra module rather than a clear error message explaining that `step_factor` must not equal 1.0.

5. **Algorithm requirement**: The finite difference algorithm fundamentally requires varying step sizes to compute derivative weights. The algorithm uses Richardson extrapolation-style nested stencils with reducing step sizes. A constant step size (when `step_factor=1.0`) breaks this core assumption.

## Relevant Context

The `scipy.differentiate.derivative` function uses an adaptive finite difference method to numerically compute derivatives. The algorithm iteratively refines its estimate by evaluating the function at different step sizes. Each iteration, the step size is divided by `step_factor` to get progressively finer approximations.

The implementation creates a Vandermonde matrix to solve for the finite difference weights. When all step sizes are identical (which happens when `step_factor=1.0`), this matrix becomes singular because all rows become linearly dependent.

Key code locations:
- Input validation: `/scipy/differentiate/_differentiate.py:31-33`
- Weights calculation: `/scipy/differentiate/_differentiate.py:678-682`
- Documentation: `/scipy/differentiate/_differentiate.py:109-114`

## Proposed Fix

```diff
--- a/scipy/differentiate/_differentiate.py
+++ b/scipy/differentiate/_differentiate.py
@@ -29,8 +29,11 @@ def _derivative_iv(f, x, args, tolerances, maxiter, order, initial_step,
                        rtol if rtol is not None else 1,
                        step_factor])
     if (not np.issubdtype(tols.dtype, np.number) or np.any(tols < 0)
-            or np.any(np.isnan(tols)) or tols.shape != (3,)):
+            or np.any(np.isnan(tols)) or tols.shape != (3,) or step_factor == 1.0):
         raise ValueError(message)
+    if step_factor == 1.0:
+        raise ValueError('`step_factor` must not equal 1.0. The algorithm requires '
+                         'varying step sizes to compute finite difference weights.')
     step_factor = float(tols[2])

     maxiter_int = int(maxiter)
```