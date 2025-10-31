# Bug Report: scipy.differentiate.derivative Accepts Invalid step_factor=0 Causing Division by Zero

**Target**: `scipy.differentiate.derivative`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `derivative` function's input validation incorrectly accepts `step_factor=0`, which causes multiple division-by-zero errors during computation, resulting in NaN outputs instead of raising a proper ValueError.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from scipy.differentiate import derivative
import numpy as np
import pytest

@given(st.just(0.0))
def test_derivative_should_reject_zero_step_factor(step_factor):
    """Test that derivative function should reject step_factor=0 with a ValueError"""
    with pytest.raises(ValueError, match="step_factor"):
        derivative(np.sin, 1.0, step_factor=step_factor)

if __name__ == "__main__":
    # Run the test
    test_derivative_should_reject_zero_step_factor()
```

<details>

<summary>
**Failing input**: `step_factor=0.0`
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/3
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_derivative_should_reject_zero_step_factor FAILED           [100%]

=================================== FAILURES ===================================
________________ test_derivative_should_reject_zero_step_factor ________________

    @given(st.just(0.0))
>   def test_derivative_should_reject_zero_step_factor(step_factor):

hypo.py:7:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

step_factor = 0.0

    @given(st.just(0.0))
    def test_derivative_should_reject_zero_step_factor(step_factor):
        """Test that derivative function should reject step_factor=0 with a ValueError"""
>       with pytest.raises(ValueError, match="step_factor"):
E       Failed: DID NOT RAISE <class 'ValueError'>
E       Falsifying example: test_derivative_should_reject_zero_step_factor(
E           step_factor=0.0,
E       )

hypo.py:9: Failed
=============================== warnings summary ===============================
hypo.py::test_derivative_should_reject_zero_step_factor
hypo.py::test_derivative_should_reject_zero_step_factor
  /home/npc/.local/lib/python3.13/site-packages/scipy/differentiate/_differentiate.py:468: RuntimeWarning: divide by zero encountered in divide
    hc = h / c**xp.arange(n, dtype=work.dtype)

hypo.py::test_derivative_should_reject_zero_step_factor
hypo.py::test_derivative_should_reject_zero_step_factor
  /home/npc/.local/lib/python3.13/site-packages/scipy/differentiate/_differentiate.py:474: RuntimeWarning: divide by zero encountered in divide
    hr = h / d**xp.arange(2*n, dtype=work.dtype)

hypo.py::test_derivative_should_reject_zero_step_factor
hypo.py::test_derivative_should_reject_zero_step_factor
  /home/npc/.local/lib/python3.13/site-packages/scipy/_lib/_elementwise_iterative_method.py:243: RuntimeWarning: invalid value encountered in sin
    f = func(x, *work.args)

hypo.py::test_derivative_should_reject_zero_step_factor
hypo.py::test_derivative_should_reject_zero_step_factor
  /home/npc/.local/lib/python3.13/site-packages/scipy/differentiate/_differentiate.py:678: RuntimeWarning: divide by zero encountered in power
    h = s / fac ** p

hypo.py::test_derivative_should_reject_zero_step_factor
hypo.py::test_derivative_should_reject_zero_step_factor
  /home/npc/.local/lib/python3.13/site-packages/scipy/differentiate/_differentiate.py:678: RuntimeWarning: divide by zero encountered in divide
    h = s / fac ** p

hypo.py::test_derivative_should_reject_zero_step_factor
hypo.py::test_derivative_should_reject_zero_step_factor
  /home/npc/.local/lib/python3.13/site-packages/scipy/differentiate/_differentiate.py:700: RuntimeWarning: divide by zero encountered in power
    h = s / np.sqrt(fac) ** p

hypo.py::test_derivative_should_reject_zero_step_factor
hypo.py::test_derivative_should_reject_zero_step_factor
  /home/npc/.local/lib/python3.13/site-packages/scipy/differentiate/_differentiate.py:700: RuntimeWarning: divide by zero encountered in divide
    h = s / np.sqrt(fac) ** p

hypo.py::test_derivative_should_reject_zero_step_factor
hypo.py::test_derivative_should_reject_zero_step_factor
  /home/npc/.local/lib/python3.13/site-packages/scipy/differentiate/_differentiate.py:542: RuntimeWarning: divide by zero encountered in divide
    work.h /= work.fac

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED hypo.py::test_derivative_should_reject_zero_step_factor - Failed: DID ...
======================== 1 failed, 16 warnings in 0.25s ========================
```
</details>

## Reproducing the Bug

```python
import numpy as np
from scipy.differentiate import derivative

# Attempt to calculate derivative with step_factor=0
# This should raise a ValueError but instead causes division by zero
result = derivative(np.sin, 1.0, step_factor=0.0)
print(f"Result: {result}")
print(f"Derivative value: {result.df}")
print(f"Success: {result.success}")
print(f"Status: {result.status}")
```

<details>

<summary>
RuntimeWarnings and NaN result instead of ValueError
</summary>
```
/home/npc/.local/lib/python3.13/site-packages/scipy/differentiate/_differentiate.py:468: RuntimeWarning: divide by zero encountered in divide
  hc = h / c**xp.arange(n, dtype=work.dtype)
/home/npc/.local/lib/python3.13/site-packages/scipy/differentiate/_differentiate.py:474: RuntimeWarning: divide by zero encountered in divide
  hr = h / d**xp.arange(2*n, dtype=work.dtype)
/home/npc/.local/lib/python3.13/site-packages/scipy/_lib/_elementwise_iterative_method.py:243: RuntimeWarning: invalid value encountered in sin
  f = func(x, *work.args)
/home/npc/.local/lib/python3.13/site-packages/scipy/differentiate/_differentiate.py:678: RuntimeWarning: divide by zero encountered in power
  h = s / fac ** p
/home/npc/.local/lib/python3.13/site-packages/scipy/differentiate/_differentiate.py:678: RuntimeWarning: divide by zero encountered in divide
  h = s / fac ** p
/home/npc/.local/lib/python3.13/site-packages/scipy/differentiate/_differentiate.py:700: RuntimeWarning: divide by zero encountered in power
  h = s / np.sqrt(fac) ** p
/home/npc/.local/lib/python3.13/site-packages/scipy/differentiate/_differentiate.py:700: RuntimeWarning: divide by zero encountered in divide
  h = s / np.sqrt(fac) ** p
/home/npc/.local/lib/python3.13/site-packages/scipy/differentiate/_differentiate.py:542: RuntimeWarning: divide by zero encountered in divide
  work.h /= work.fac
Result:      success: False
      status: -3
          df: nan
       error: nan
         nit: 1
        nfev: 9
           x: 1.0
Derivative value: nan
Success: False
Status: -3
```
</details>

## Why This Is A Bug

The validation logic at lines 27-33 of `scipy/differentiate/_differentiate.py` incorrectly allows `step_factor=0` to pass validation. The current check `np.any(tols < 0)` only rejects negative values, accepting zero since `0 >= 0`. However, the algorithm mathematically requires `step_factor > 0` for the following reasons:

1. **Division by zero in step calculation**: At line 468, the code computes `hc = h / c**xp.arange(n, dtype=work.dtype)` where `c = step_factor`. When `c=0`, this produces `c**1 = 0`, `c**2 = 0`, etc., leading to division by zero for all terms except the first.

2. **Mathematical interpretation**: The documentation describes `step_factor` as "the factor by which the step size is *reduced*" with the formula `new_step = initial_step / step_factor`. A zero reduction factor has no valid mathematical meaning and results in undefined behavior.

3. **Inconsistent error messaging**: The error message states "Tolerances and step parameters must be non-negative scalars" which implies `>= 0` is acceptable, but the algorithm requires strictly positive values for `step_factor`.

4. **Confusing user experience**: Instead of getting a clear validation error, users encounter multiple RuntimeWarnings and receive a result with `success=False`, `status=-3` (non-finite value encountered), and `df=nan`.

## Relevant Context

The finite difference algorithm in `scipy.differentiate.derivative` uses Richardson extrapolation with progressively smaller step sizes. The `step_factor` parameter controls how quickly these steps decrease. The algorithm is based on the following mathematical principle:

- In iteration `i`, the step size is `h_i = initial_step / (step_factor^i)`
- When `step_factor=0`, all iterations after the first have undefined step sizes
- The algorithm also uses `sqrt(step_factor)` for one-sided differences, which becomes 0

The current validation groups `step_factor` with tolerance parameters that can legitimately be zero, but `step_factor` has different mathematical requirements and should be validated separately.

Documentation link: https://docs.scipy.org/doc/scipy/reference/generated/scipy.differentiate.derivative.html
Source code: https://github.com/scipy/scipy/blob/main/scipy/differentiate/_differentiate.py

## Proposed Fix

```diff
--- a/scipy/differentiate/_differentiate.py
+++ b/scipy/differentiate/_differentiate.py
@@ -24,12 +24,16 @@ def _derivative_iv(f, x, args, tolerances, maxiter, order, initial_step,
     rtol = tolerances.get('rtol', None)

     # tolerances are floats, not arrays; OK to use NumPy
-    message = 'Tolerances and step parameters must be non-negative scalars.'
+    message = 'Tolerances must be non-negative scalars.'
     tols = np.asarray([atol if atol is not None else 1,
-                       rtol if rtol is not None else 1,
-                       step_factor])
-    if (not np.issubdtype(tols.dtype, np.number) or np.any(tols < 0)
-            or np.any(np.isnan(tols)) or tols.shape != (3,)):
+                       rtol if rtol is not None else 1])
+    if (not np.issubdtype(tols.dtype, np.number) or np.any(tols < 0)
+            or np.any(np.isnan(tols)) or tols.shape != (2,)):
+        raise ValueError(message)
+
+    # step_factor must be strictly positive
+    if (not np.isscalar(step_factor) or step_factor <= 0
+            or np.isnan(step_factor)):
+        raise ValueError('`step_factor` must be a positive scalar.')
-    step_factor = float(tols[2])
+    step_factor = float(step_factor)
```