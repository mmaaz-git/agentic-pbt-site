# Bug Report: scipy.optimize.bisect Interval Direction Non-Determinism

**Target**: `scipy.optimize.bisect`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `bisect` root-finding function returns different roots when the interval endpoints are swapped, exhibiting non-deterministic behavior at specific floating-point boundary values.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume, settings
from scipy.optimize import bisect
import math


@given(
    st.floats(min_value=-10, max_value=-0.1, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.1, max_value=10, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=500)
def test_interval_direction_invariance(a, b):
    assume(abs(a - b) > 0.5)

    def f(x):
        return x**3 - x

    fa, fb = f(a), f(b)
    assume(fa * fb < 0)

    root_ab = bisect(f, a, b)
    root_ba = bisect(f, b, a)

    assert math.isclose(root_ab, root_ba, rel_tol=1e-9, abs_tol=1e-9), \
        f"bisect gave different results: f({a},{b})={root_ab} vs f({b},{a})={root_ba}"
```

**Failing input**: `a=-10.0`, `b=7.999999999999998`

## Reproducing the Bug

```python
from scipy.optimize import bisect


def f(x):
    return x**3 - x


a, b = -10.0, 7.999999999999998

root_ab = bisect(f, a, b)
root_ba = bisect(f, b, a)

print(f"bisect(f, {a}, {b}) = {root_ab}")
print(f"bisect(f, {b}, {a}) = {root_ba}")
print(f"Difference: {abs(root_ab - root_ba)}")
```

Output:
```
bisect(f, -10.0, 7.999999999999998) = -1.0
bisect(f, 7.999999999999998, -10.0) = 1.0000000000005667
Difference: 2.0000000000005667
```

## Why This Is A Bug

The bisection algorithm should be deterministic with respect to interval direction. While the function `f(x) = x³ - x` has three roots in the interval `[-10, 8]` (at x = -1, 0, 1), and finding any valid root is acceptable, the behavior should be consistent regardless of whether the interval is specified as `[a, b]` or `[b, a]`.

The documentation describes `a` and `b` as "One end of the bracketing interval [a,b]" and "The other end of the bracketing interval [a,b]", suggesting they should be interchangeable. However, the implementation exhibits non-deterministic behavior at specific floating-point boundaries.

This bug only affects `bisect`. Other root-finding methods (`brentq`, `brenth`, `ridder`) correctly return the same root regardless of interval direction.

## Fix

The issue likely stems from how the C implementation handles interval endpoints when `a > b`. The algorithm should either:

1. Normalize the interval at the start by swapping endpoints if necessary to ensure `a < b`
2. Document that `a` must be less than `b` and raise an error otherwise

Recommended approach: Add interval normalization in the C implementation `_zeros._bisect`:

```diff
--- a/scipy/optimize/_zeros.pyx
+++ b/scipy/optimize/_zeros.pyx
@@ bisect function
+    # Normalize interval to ensure a < b
+    if a > b:
+        a, b = b, a
+
     # Continue with existing bisection logic
```

Alternatively, if normalization is undesirable, the Python wrapper should validate and document the requirement:

```diff
--- a/scipy/optimize/_zeros_py.py
+++ b/scipy/optimize/_zeros_py.py
@@ -587,6 +587,8 @@ def bisect(f, a, b, args=(),
     if not isinstance(args, tuple):
         args = (args,)
     maxiter = operator.index(maxiter)
+    if a >= b:
+        raise ValueError(f"a must be less than b, but a={a} >= b={b}")
     if xtol <= 0:
         raise ValueError(f"xtol too small ({xtol:g} <= 0)")
```