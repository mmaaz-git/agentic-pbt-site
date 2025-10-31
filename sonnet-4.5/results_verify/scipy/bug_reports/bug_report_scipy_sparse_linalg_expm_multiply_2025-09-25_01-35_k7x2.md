# Bug Report: scipy.sparse.linalg.expm_multiply ZeroDivisionError

**Target**: `scipy.sparse.linalg.expm_multiply`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `expm_multiply` function crashes with `ZeroDivisionError` when called with extremely small time intervals (near machine epsilon). The bug occurs in the internal time-stepping logic when the scaling factor `s` becomes zero.

## Property-Based Test

```python
@settings(max_examples=200)
@given(
    n=st.integers(min_value=2, max_value=20),
    t1=st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    t2=st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    seed=st.integers(min_value=0, max_value=10000)
)
def test_expm_multiply_semigroup(n, t1, t2, seed):
    np.random.seed(seed)
    A = sp.random(n, n, density=0.3, format='csr', dtype=float) * 0.5
    v = np.random.randn(n)

    result1 = sla.expm_multiply(A, v, start=0, stop=t1 + t2, num=2, endpoint=True)[-1]
    intermediate = sla.expm_multiply(A, v, start=0, stop=t1, num=2, endpoint=True)[-1]
    result2 = sla.expm_multiply(A, intermediate, start=0, stop=t2, num=2, endpoint=True)[-1]

    relative_error = np.linalg.norm(result1 - result2) / (np.linalg.norm(result1) + 1e-10)
    assert relative_error < 1e-4
```

**Failing input**: `n=3, t1=0.0, t2=5e-324, seed=0`

## Reproducing the Bug

```python
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

np.random.seed(0)
n = 3
A = sp.random(n, n, density=0.3, format='csr', dtype=float) * 0.5
v = np.random.randn(n)

result = sla.expm_multiply(A, v, start=0, stop=5e-324, num=2, endpoint=True)
```

**Error:**
```
ZeroDivisionError: integer modulo by zero
  File "scipy/sparse/linalg/_expm_multiply.py", line 719, in _expm_multiply_interval
    elif not (q % s):
```

## Why This Is A Bug

The function should handle all valid floating-point time intervals, including very small ones. The crash occurs because:

1. When computing the scaling factor `s` in `_fragment_3_1()`, if the 1-norm of certain matrix computations is zero, `s` can become 0
2. At line 719, the code performs `q % s` without checking if `s` is zero
3. The safeguard `s = max(s, 1)` exists in some code paths but not in Path 1 (line 547) where `s = int(np.ceil(norm_info.onenorm() / theta))`

This violates the documented behavior that `expm_multiply` should compute the matrix exponential for valid inputs, and crashes instead.

## Fix

The root cause is in the `_fragment_3_1` function where the scaling factor `s` can become zero in Path 1. The fix is to add a safeguard similar to Path 2:

```diff
--- a/scipy/sparse/linalg/_expm_multiply.py
+++ b/scipy/sparse/linalg/_expm_multiply.py
@@ -545,6 +545,7 @@ def _fragment_3_1(norm_info, n0, tol, m_max=55, ell=2):
         for p in range(2, p_max+1):
             best_m = _compute_cost_div_m(m, p, norm_info)
             best_s = int(np.ceil(norm_info.onenorm() / _theta[best_m]))
+            best_s = max(best_s, 1)  # Ensure s is never zero
             cost = best_m * best_s
             if cost < best_cost:
                 best_cost = cost
```

Alternatively, add a check before the modulo operation at line 719:

```diff
--- a/scipy/sparse/linalg/_expm_multiply.py
+++ b/scipy/sparse/linalg/_expm_multiply.py
@@ -716,6 +716,8 @@ def _expm_multiply_interval(A, B, start, stop, num,
         return _expm_multiply_interval_core_0(A, X, h, mu, q, norm_info, tol, ell, n0)
 elif not (q % s):
+    if s == 0:
+        raise ValueError(f"Invalid scaling factor s=0, likely due to degenerate matrix")
     if status_only:
         return 1
```

The first fix (ensuring `s >= 1` in `_fragment_3_1`) is preferable as it addresses the root cause rather than just catching the symptom.