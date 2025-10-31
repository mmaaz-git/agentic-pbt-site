# Bug Report: scipy.integrate.tanhsinh IndexError on Constant Functions

**Target**: `scipy.integrate.tanhsinh`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `scipy.integrate.tanhsinh` function crashes with an IndexError when integrating any constant function, regardless of the constant value or integration limits. This occurs because the function incorrectly handles scalar returns from non-vectorized functions.

## Property-Based Test

```python
import math
import numpy as np
from hypothesis import given, strategies as st, assume, settings
from scipy.integrate import tanhsinh

@given(
    st.floats(allow_nan=False, allow_infinity=False, min_value=-1e3, max_value=1e3),
    st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100),
    st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100)
)
@settings(max_examples=300)
def test_tanhsinh_constant(c, a, b):
    assume(abs(b - a) > 0.01)
    assume(abs(a) < 100 and abs(b) < 100)

    def f(x):
        return c

    result = tanhsinh(f, a, b)
    expected = c * (b - a)

    assert math.isclose(result.integral, expected, rel_tol=1e-8, abs_tol=1e-10)

if __name__ == "__main__":
    test_tanhsinh_constant()
```

<details>

<summary>
**Failing input**: `c=0.0, a=0.0, b=1.0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/55/hypo.py", line 25, in <module>
    test_tanhsinh_constant()
    ~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/55/hypo.py", line 7, in test_tanhsinh_constant
    st.floats(allow_nan=False, allow_infinity=False, min_value=-1e3, max_value=1e3),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/55/hypo.py", line 19, in test_tanhsinh_constant
    result = tanhsinh(f, a, b)
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/integrate/_tanhsinh.py", line 492, in tanhsinh
    res = eim._loop(work, callback, shape, maxiter, f, args, dtype, pre_func_eval,
                    post_func_eval, check_termination, post_termination_check,
                    customize_result, res_work_pairs, xp, preserve_shape)
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/_lib/_elementwise_iterative_method.py", line 250, in _loop
    post_func_eval(x, f, work)
    ~~~~~~~~~~~~~~^^^^^^^^^^^^
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/integrate/_tanhsinh.py", line 421, in post_func_eval
    fj[work.abinf] *= ((1 + work.xj[work.abinf]**2) /
    ~~^^^^^^^^^^^^
IndexError: too many indices for array: array is 0-dimensional, but 1 were indexed
Falsifying example: test_tanhsinh_constant(
    # The test always failed when commented parts were varied together.
    c=0.0,  # or any other generated value
    a=0.0,  # or any other generated value
    b=1.0,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
from scipy.integrate import tanhsinh

def f(x):
    return 1.0

result = tanhsinh(f, 0.0, 1.0)
print(f"Integral result: {result.integral}")
```

<details>

<summary>
IndexError: too many indices for array
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/55/repo.py", line 6, in <module>
    result = tanhsinh(f, 0.0, 1.0)
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/integrate/_tanhsinh.py", line 492, in tanhsinh
    res = eim._loop(work, callback, shape, maxiter, f, args, dtype, pre_func_eval,
                    post_func_eval, check_termination, post_termination_check,
                    customize_result, res_work_pairs, xp, preserve_shape)
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/_lib/_elementwise_iterative_method.py", line 250, in _loop
    post_func_eval(x, f, work)
    ~~~~~~~~~~~~~~^^^^^^^^^^^^
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/integrate/_tanhsinh.py", line 421, in post_func_eval
    fj[work.abinf] *= ((1 + work.xj[work.abinf]**2) /
    ~~^^^^^^^^^^^^
IndexError: too many indices for array: array is 0-dimensional, but 1 were indexed
```
</details>

## Why This Is A Bug

The documentation states that `f` must be an "elementwise function" but doesn't clearly define that this requires the function to return arrays with the same shape as the input. The documentation signature shows `f(xi: ndarray, *argsi) -> ndarray`, but many Python users naturally write constant functions as `lambda x: c`, which returns a scalar.

The bug occurs because:
1. The integrator calls `f` with array inputs of varying shapes (e.g., scalar shape `()` and then `(1, 66)`)
2. When a constant function returns a scalar (0-dimensional array), the code at line 421 attempts boolean indexing `fj[work.abinf]`
3. Boolean indexing on a 0-dimensional array fails with IndexError

Integrating constant functions is fundamental - the integral of constant `c` from `a` to `b` is simply `c * (b - a)`. This is one of the first integrals taught in calculus. The fact that scipy's integration routine crashes on this basic case violates mathematical expectations and contradicts the principle of least surprise.

## Relevant Context

Testing reveals that the function is called with different input shapes during integration:
- First call: scalar input with shape `()`
- Second call: 2D array input with shape `(1, 66)`

Functions that work correctly are those that preserve input shape:
- ✓ `lambda x: np.ones_like(x)` - returns array matching input shape
- ✓ `lambda x: np.full_like(x, 1.0)` - returns array matching input shape
- ✗ `lambda x: 1.0` - returns scalar regardless of input shape
- ✗ `lambda x: c` for any constant `c` - returns scalar

The documentation link: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.tanhsinh.html

## Proposed Fix

```diff
--- a/scipy/integrate/_tanhsinh.py
+++ b/scipy/integrate/_tanhsinh.py
@@ -413,6 +413,10 @@

     def post_func_eval(x, fj, work):
         # Weight integrand as required by substitutions for infinite limits
+        # Ensure fj is at least 1-D to support indexing operations
+        if fj.ndim == 0:
+            fj = xp.atleast_1d(fj)
+
         if work.log:
             fj[work.abinf] += (xp.log(1 + work.xj[work.abinf]**2)
                                - 2*xp.log(1 - work.xj[work.abinf]**2))
```