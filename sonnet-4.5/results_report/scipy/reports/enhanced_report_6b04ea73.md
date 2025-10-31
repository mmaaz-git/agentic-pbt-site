# Bug Report: scipy.integrate.tanhsinh Crashes with Scalar-Returning Functions

**Target**: `scipy.integrate.tanhsinh`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`scipy.integrate.tanhsinh` crashes with an `IndexError` when passed a function that returns a scalar instead of an array, while other scipy integration functions handle scalar-returning functions correctly.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, assume
import numpy as np
from scipy import integrate

@given(
    k=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    x_min=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
    x_max=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=500)
def test_tanhsinh_constant(k, x_min, x_max):
    assume(x_min < x_max)
    result = integrate.tanhsinh(lambda x: k, x_min, x_max)
    expected = k * (x_max - x_min)
    assert np.isclose(result.integral, expected, rtol=1e-10)

# Run the test
test_tanhsinh_constant()
```

<details>

<summary>
**Failing input**: `k=0.0, x_min=1e-10, x_max=51.99428321165712`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/4/hypo.py", line 18, in <module>
    test_tanhsinh_constant()
    ~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/4/hypo.py", line 6, in test_tanhsinh_constant
    k=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/4/hypo.py", line 13, in test_tanhsinh_constant
    result = integrate.tanhsinh(lambda x: k, x_min, x_max)
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
    k=0.0,
    x_min=1e-10,
    x_max=51.99428321165712,
)
```
</details>

## Reproducing the Bug

```python
from scipy import integrate

# Test case that should crash with IndexError
result = integrate.tanhsinh(lambda x: 0.0, 0.0, 1.0)
print(f"Result: {result}")
```

<details>

<summary>
IndexError: too many indices for array
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/4/repo.py", line 4, in <module>
    result = integrate.tanhsinh(lambda x: 0.0, 0.0, 1.0)
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

This violates expected behavior for several critical reasons:

1. **API Inconsistency**: Other scipy integration functions handle scalar-returning functions correctly:
   - `integrate.quad(lambda x: 0.0, 0.0, 1.0)` returns `(0.0, 0.0)`
   - `integrate.quad_vec(lambda x: 0.0, 0.0, 1.0)` returns `(0.0, 0.0)`
   - Only `integrate.tanhsinh(lambda x: 0.0, 0.0, 1.0)` crashes with an IndexError

2. **Documentation vs Implementation Gap**: While the documentation specifies `f(xi: ndarray, *argsi) -> ndarray`, the actual failure mode is problematic:
   - The documentation states the function must be "elementwise" and handle arrays
   - However, when a user provides a mathematically valid constant function, it crashes deep in the implementation
   - The error message (`IndexError: too many indices for array`) provides no guidance about the actual issue

3. **Common Mathematical Usage**: Writing `lambda x: c` for a constant function is standard mathematical notation that users naturally expect to work for integration

4. **Poor Error Handling**: Instead of detecting the scalar return and either:
   - Converting it to the expected array format automatically
   - Providing a clear error message about the function signature requirement
   The function crashes with a cryptic IndexError at line 421 in `_tanhsinh.py`

## Relevant Context

The crash occurs in the `post_func_eval` function in `/home/npc/.local/lib/python3.13/site-packages/scipy/integrate/_tanhsinh.py` at line 421:

```python
fj[work.abinf] *= ((1 + work.xj[work.abinf]**2) / (1 - work.xj[work.abinf]**2)**2)
```

Here, `fj` is expected to be an array that can be indexed with `work.abinf` (a boolean mask), but when the user's function returns a scalar, `fj` is 0-dimensional and cannot be indexed.

**Workaround**: Users can work around this by ensuring their function returns an array:
```python
integrate.tanhsinh(lambda x: np.full_like(x, 0.0), 0.0, 1.0)  # Works
```

However, this workaround is non-intuitive and not documented.

**Documentation Reference**: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.tanhsinh.html

## Proposed Fix

```diff
--- a/scipy/integrate/_tanhsinh.py
+++ b/scipy/integrate/_tanhsinh.py
@@ -413,12 +413,22 @@ def tanhsinh(f, a, b, *, args=(), log=False, maxlevel=None, minlevel=2,

     def post_func_eval(x, fj, work):
         # Weight integrand as required by substitutions for infinite limits
+
+        # Handle scalar returns by broadcasting to expected shape
+        if fj.ndim == 0 and work.xj.ndim > 0:
+            fj = xp.broadcast_to(fj, work.xj.shape)
+
         if work.log:
-            fj[work.abinf] += (xp.log(1 + work.xj[work.abinf]**2)
-                              - 2*xp.log(1 - work.xj[work.abinf]**2))
-            fj[work.binf] -= 2 * xp.log(work.xj[work.binf])
+            fj_copy = xp_copy(fj)
+            fj_copy[work.abinf] += (xp.log(1 + work.xj[work.abinf]**2)
+                                   - 2*xp.log(1 - work.xj[work.abinf]**2))
+            fj_copy[work.binf] -= 2 * xp.log(work.xj[work.binf])
+            fj = fj_copy
         else:
-            fj[work.abinf] *= ((1 + work.xj[work.abinf]**2) /
-                               (1 - work.xj[work.abinf]**2)**2)
-            fj[work.binf] *= work.xj[work.binf]**-2.
+            fj_copy = xp_copy(fj)
+            fj_copy[work.abinf] *= ((1 + work.xj[work.abinf]**2) /
+                                   (1 - work.xj[work.abinf]**2)**2)
+            fj_copy[work.binf] *= work.xj[work.binf]**-2.
+            fj = fj_copy

         # Estimate integral with Euler-Maclaurin Sum
```