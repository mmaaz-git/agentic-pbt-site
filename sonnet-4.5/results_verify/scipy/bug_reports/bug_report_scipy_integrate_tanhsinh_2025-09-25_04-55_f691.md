# Bug Report: scipy.integrate.tanhsinh IndexError on Constant Functions

**Target**: `scipy.integrate.tanhsinh`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `tanhsinh` function crashes with an `IndexError` when attempting to integrate any constant function, regardless of integration limits or constant value. This is a fundamental use case that should work correctly.

## Property-Based Test

```python
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
```

**Failing input**: `c=0.0, a=0.0, b=1.0` (and all other constant function inputs)

## Reproducing the Bug

```python
from scipy.integrate import tanhsinh

def f(x):
    return 1.0

result = tanhsinh(f, 0.0, 1.0)
```

Output:
```
IndexError: too many indices for array: array is 0-dimensional, but 1 were indexed
```

Full traceback:
```
Traceback (most recent call last):
  File "reproduce_tanhsinh_bug.py", line 14, in <module>
    result = tanhsinh(f, 0.0, 1.0)
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/integrate/_tanhsinh.py", line 492, in tanhsinh
    res = eim._loop(work, callback, shape, maxiter, f, args, dtype, pre_func_eval,
                    post_func_eval, check_termination, post_termination_check,
                    customize_result, res_work_pairs, xp, preserve_shape)
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/_lib/_elementwise_iterative_method.py", line 250, in _loop
    post_func_eval(x, f, work)
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/integrate/_tanhsinh.py", line 421, in post_func_eval
    fj[work.abinf] *= ((1 + work.xj[work.abinf]**2) /
IndexError: too many indices for array: array is 0-dimensional, but 1 were indexed
```

## Why This Is A Bug

Integrating a constant function is one of the most basic operations in numerical integration. The integral of a constant `c` from `a` to `b` should always equal `c * (b - a)`.

Testing shows that:
- ✗ All constant functions crash: `f(x) = 1.0`, `f(x) = 5.0`, `f(x) = -3.0`
- ✗ Crashes regardless of integration limits: `[0,1]`, `[-1,1]`, `[1,2]`
- ✓ Non-constant functions work: `f(x) = x`, `f(x) = x**2`

The error occurs in `_tanhsinh.py` at line 421 in the `post_func_eval` function, where the code tries to index `fj` as an array when it is a 0-dimensional array (scalar) for constant functions.

## Fix

The root cause is that when a function returns a constant value, the result `fj` becomes a 0-dimensional numpy array (scalar), but the code at line 421 attempts to use boolean indexing on it:

```python
fj[work.abinf] *= ((1 + work.xj[work.abinf]**2) / ...
```

The fix should handle the case when `fj` is a scalar (0-dimensional array). This requires checking the dimensionality of `fj` before attempting to index it, or ensuring `fj` is always at least 1-dimensional.

A potential fix would be to modify the `post_func_eval` function to handle scalar results:

```python
# Before indexing, ensure fj is at least 1-D
if fj.ndim == 0:
    fj = np.atleast_1d(fj)
```

Or more robustly, check if indexing is needed at all:

```python
# Only apply transformation if there are infinite bounds
if work.abinf.any() if hasattr(work.abinf, 'any') else work.abinf:
    if fj.ndim == 0:
        fj = np.atleast_1d(fj)
    fj[work.abinf] *= ...
```

The exact fix would require examining the full context of how `fj` and `work.abinf` are used throughout the function to ensure dimensional consistency.