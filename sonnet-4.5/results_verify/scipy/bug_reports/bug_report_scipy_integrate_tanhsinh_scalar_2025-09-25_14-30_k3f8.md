# Bug Report: scipy.integrate.tanhsinh Crashes with Scalar-Returning Functions

**Target**: `scipy.integrate.tanhsinh`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`scipy.integrate.tanhsinh` crashes with an `IndexError` when passed a function that returns a scalar instead of an array, while other scipy integration functions (`quad`, `quad_vec`) handle scalar-returning functions correctly.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
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
```

**Failing input**: `k=0.0, x_min=0.0, x_max=1.0`

## Reproducing the Bug

```python
from scipy import integrate

result = integrate.tanhsinh(lambda x: 0.0, 0.0, 1.0)
```

Error:
```
IndexError: too many indices for array: array is 0-dimensional, but 1 were indexed
```

Full traceback:
```
File "/home/npc/.local/lib/python3.13/site-packages/scipy/integrate/_tanhsinh.py", line 492, in tanhsinh
    res = eim._loop(work, callback, shape, maxiter, f, args, dtype, pre_func_eval,
                    post_func_eval, check_termination, post_termination_check,
                    customize_result, res_work_pairs, xp, preserve_shape)
File "/home/npc/.local/lib/python3.13/site-packages/scipy/_lib/_elementwise_iterative_method.py", line 250, in _loop
    post_func_eval(x, f, work)
File "/home/npc/.local/lib/python3.13/site-packages/scipy/integrate/_tanhsinh.py", line 421, in post_func_eval
    fj[work.abinf] *= ((1 + work.xj[work.abinf]**2) /
    ~~^^^^^^^^^^^^
IndexError: too many indices for array: array is 0-dimensional, but 1 were indexed
```

## Why This Is A Bug

1. **Inconsistency**: Other scipy integration functions (`quad`, `quad_vec`) handle scalar-returning functions correctly:
   ```python
   from scipy import integrate

   integrate.quad(lambda x: 0.0, 0.0, 1.0)  # Works: (0.0, 0.0)
   integrate.quad_vec(lambda x: 0.0, 0.0, 1.0)  # Works: (0.0, 0.0)
   integrate.tanhsinh(lambda x: 0.0, 0.0, 1.0)  # Crashes: IndexError
   ```

2. **Common usage pattern**: Users naturally write simple scalar-returning functions like `lambda x: x**2` for integration, which is standard mathematical notation.

3. **Poor error handling**: Instead of providing a clear error message about expected function signature, the function crashes deep in the implementation with a confusing IndexError.

4. **Workaround exists but is non-obvious**: The function works if users wrap the scalar in `np.full_like(x, value)`, but this is not documented or intuitive:
   ```python
   integrate.tanhsinh(lambda x: np.full_like(x, 0.0), 0.0, 1.0)  # Works
   ```

## Fix

The bug is in `_tanhsinh.py` at line 421 in the `post_func_eval` function. The code attempts to index `fj` with a mask `work.abinf`, but when the user's function returns a scalar, `fj` is 0-dimensional and cannot be indexed.

The fix should handle the case where `fj` is a scalar by converting it to an array with the appropriate shape before indexing. A patch might look like:

```diff
--- a/scipy/integrate/_tanhsinh.py
+++ b/scipy/integrate/_tanhsinh.py
@@ -415,10 +415,14 @@ def tanhsinh(f, a, b, *, args=(), log=False, maxlevel=None, minlevel=2,
         # Transform the function values as needed for infinite bounds
         # The transformation is fj *= ((1 + xj**2) / (2 * tj))
         # where the condition is checked by work.abinf
+
+        # Ensure fj is at least 1-dimensional to support indexing
+        fj_orig_ndim = fj.ndim
+        if fj.ndim == 0:
+            fj = fj[np.newaxis]
+
         fj[work.abinf] *= ((1 + work.xj[work.abinf]**2) /
                             (2 * work.tj[work.abinf]))
-
-        work.fj = fj
+        work.fj = fj if fj_orig_ndim > 0 else fj[0]
```

Alternatively, the function could detect scalar returns early and wrap them appropriately, or provide a clear error message directing users to use array-returning functions.