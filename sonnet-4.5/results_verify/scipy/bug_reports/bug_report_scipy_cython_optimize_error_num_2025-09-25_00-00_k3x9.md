# Bug Report: scipy.optimize.cython_optimize Undocumented Negative Error Code

**Target**: `scipy.optimize.cython_optimize._zeros.full_output_example`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `full_output_example` function returns an undocumented negative error code (`error_num = -2`) when the root-finding algorithm fails to converge within the maximum iterations, violating API contract expectations and differing from the behavior of scipy's standard optimize functions.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, assume
from scipy.optimize.cython_optimize import _zeros

def polynomial_f(x, args):
    a0, a1, a2, a3 = args
    return ((a3 * x + a2) * x + a1) * x + a0

@given(
    a0=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
    a1=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
    a2=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
    a3=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
    xa=st.floats(min_value=-100, max_value=0, allow_nan=False, allow_infinity=False),
    xb=st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False),
    mitr=st.integers(min_value=10, max_value=1000),
)
@settings(max_examples=500)
def test_output_validity(a0, a1, a2, a3, xa, xb, mitr):
    assume(xa < xb)

    args = (a0, a1, a2, a3)
    f_xa = polynomial_f(xa, args)
    f_xb = polynomial_f(xb, args)

    assume(f_xa * f_xb < 0)

    xtol, rtol = 1e-6, 1e-6

    output = _zeros.full_output_example(args, xa, xb, xtol, rtol, mitr)

    assert output['error_num'] >= 0, f"error_num should be non-negative, got {output['error_num']}"
```

**Failing input**: `a0=0.0, a1=0.0, a2=0.0, a3=1.0, xa=-1.0, xb=2.0, mitr=10`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/scipy_env')

from scipy.optimize.cython_optimize import _zeros

args = (0.0, 0.0, 0.0, 1.0)
xa, xb = -1.0, 2.0
xtol, rtol, mitr = 1e-6, 1e-6, 10

output = _zeros.full_output_example(args, xa, xb, xtol, rtol, mitr)

print(f"error_num: {output['error_num']}")
print(f"Output: {output}")

assert output['error_num'] >= 0, f"error_num is negative: {output['error_num']}"
```

Output:
```
error_num: -2
Output: {'funcalls': 12, 'iterations': 10, 'error_num': -2, 'root': -0.08503582770913126}
AssertionError: error_num is negative: -2
```

## Why This Is A Bug

1. **Undocumented behavior**: The function's docstring states it returns "the zero function error number" but provides no documentation about what error codes mean or that they can be negative.

2. **Contract violation**: Error codes in most APIs are non-negative integers by convention (0 = success, positive = error types). Negative error codes are unexpected and undocumented.

3. **Inconsistency with scipy**: The standard `scipy.optimize.brentq` function raises a `RuntimeError` when convergence fails, rather than returning an error code:
   ```python
   from scipy.optimize import brentq
   brentq(lambda x: x**3, -1.0, 2.0, xtol=1e-6, rtol=1e-6, maxiter=10, full_output=True)
   # Raises: RuntimeError: Failed to converge after 10 iterations.
   ```

4. **Silent failure**: Users calling this function have no way to know what `error_num = -2` means, and may not check the error code at all, leading to incorrect results being used.

## Fix

The function should either:
1. Document the meaning of all possible error codes (including -2), or
2. Raise an exception like the standard scipy.optimize functions, or
3. Change negative error codes to positive values and document them

Since this is part of the Cython API and may be used by external code, option 1 (documentation) is the safest fix:

```diff
diff --git a/scipy/optimize/cython_optimize/_zeros.pyx b/scipy/optimize/cython_optimize/_zeros.pyx
--- a/scipy/optimize/cython_optimize/_zeros.pyx
+++ b/scipy/optimize/cython_optimize/_zeros.pyx
@@ -XX,X +XX,X @@ def full_output_example(args, xa, xb, xtol, rtol, mitr):
     Returns
     -------
     full_output : dict
         the root, number of function calls, number of iterations, and the zero
-        function error number
+        function error number. Error codes:
+        - 0: Convergence successful
+        - -2: Failed to converge within maximum iterations
```

Alternatively, for better consistency with scipy, raise an exception when error_num != 0.