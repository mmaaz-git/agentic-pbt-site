# Bug Report: scipy.differentiate.derivative Singular Matrix Crash with step_factor=1.0

**Target**: `scipy.differentiate.derivative`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `derivative` function crashes with `numpy.linalg.LinAlgError: Singular matrix` when `step_factor` is set to 1.0, even though the documentation and input validation do not prohibit this value.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
from scipy.differentiate import derivative

@given(x=st.floats(min_value=0.5, max_value=2, allow_nan=False, allow_infinity=False))
@settings(max_examples=100)
def test_step_factor_exactly_one(x):
    def f(x_val):
        return x_val ** 2

    res = derivative(f, x, step_factor=1.0, maxiter=5)

    if res.success:
        assert abs(res.df - 2 * x) < 1e-6

test_step_factor_exactly_one()
```

<details>

<summary>
**Failing input**: `x=1.0` (or any value)
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/43/hypo.py", line 16, in <module>
    test_step_factor_exactly_one()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/43/hypo.py", line 6, in test_step_factor_exactly_one
    @settings(max_examples=100)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/43/hypo.py", line 11, in test_step_factor_exactly_one
    res = derivative(f, x, step_factor=1.0, maxiter=5)
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
Falsifying example: test_step_factor_exactly_one(
    x=1.0,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
from scipy.differentiate import derivative

def f(x_val):
    return x_val ** 2

x = 1.0

try:
    res = derivative(f, x, step_factor=1.0, maxiter=2)
    print(f"Result: {res.df}")
except np.linalg.LinAlgError as e:
    print(f"Crash: {e}")
```

<details>

<summary>
Output shows singular matrix error
</summary>
```
Crash: Singular matrix
```
</details>

## Why This Is A Bug

The function crashes with an unhandled `LinAlgError` when `step_factor=1.0` due to a mathematical degeneracy in the finite difference weight calculation. Here's exactly why this violates expected behavior:

1. **Documentation allows this value**: The `step_factor` parameter documentation (lines 109-114 in `_differentiate.py`) states: "The factor by which the step size is *reduced* in each iteration" and explicitly discusses values < 1, but never mentions that 1.0 is invalid or will cause a crash.

2. **Input validation misses this case**: The `_derivative_iv` function (lines 28-33) validates that `step_factor >= 0` but doesn't check for equality to 1.0, allowing this problematic value to pass through.

3. **Mathematical root cause**: In the `_derivative_weights` function (lines 678-682 for central differences, 700-704 for one-sided), the code computes `h = s / fac ** p` where `fac` is the `step_factor`. When `step_factor=1.0`:
   - All powers become identical: `1.0**p = 1.0` for any `p`
   - This creates duplicate rows in the Vandermonde matrix `A`
   - The matrix becomes singular (non-invertible)
   - `np.linalg.solve(A, b)` raises `LinAlgError`

4. **Poor user experience**: Users encounter an obscure linear algebra error rather than a clear validation message explaining that `step_factor` must not equal 1.0.

5. **Valid edge case handling**: While `step_factor=1.0` is mathematically nonsensical (no step reduction occurs between iterations), good software should either validate against it or handle it gracefully.

## Relevant Context

The finite difference method implemented in `scipy.differentiate.derivative` relies on evaluating the function at multiple points with progressively smaller step sizes. Each iteration reduces the step by `step_factor`. When `step_factor=1.0`, the step size never changes, making all evaluations at the same points, which breaks the mathematical foundation of the Richardson extrapolation-like approach used.

The code already has infrastructure for input validation but fails to catch this specific edge case. Values very close to 1.0 (e.g., 0.9999 or 1.0001) work fine, indicating this is specifically about the exact value 1.0.

Documentation reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.differentiate.derivative.html

Code location: `/home/npc/.local/lib/python3.13/site-packages/scipy/differentiate/_differentiate.py`

## Proposed Fix

Add validation to reject `step_factor == 1.0` in the `_derivative_iv` function:

```diff
--- a/scipy/differentiate/_differentiate.py
+++ b/scipy/differentiate/_differentiate.py
@@ -28,6 +28,9 @@ def _derivative_iv(f, x, args, tolerances, maxiter, order, initial_step,
     tols = np.asarray([atol if atol is not None else 1,
                        rtol if rtol is not None else 1,
                        step_factor])
+    if step_factor == 1.0:
+        raise ValueError('`step_factor` must not equal 1.0. The step size must '
+                         'change between iterations for the algorithm to work.')
     if (not np.issubdtype(tols.dtype, np.number) or np.any(tols < 0)
             or np.any(np.isnan(tols)) or tols.shape != (3,)):
         raise ValueError(message)
```