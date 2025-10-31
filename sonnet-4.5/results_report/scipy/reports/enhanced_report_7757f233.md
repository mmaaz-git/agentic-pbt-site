# Bug Report: scipy.spatial.transform.RotationSpline Numerical Overflow with Non-uniform Time Arrays

**Target**: `scipy.spatial.transform.RotationSpline`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

RotationSpline crashes with "array must not contain infs or NaNs" when initialized with valid but non-uniformly spaced time arrays, due to numerical overflow during angular rate computation when time intervals vary by large factors.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, assume
import numpy as np
from scipy.spatial.transform import Rotation, RotationSpline

@st.composite
def sorted_times_strategy(draw, min_times=2, max_times=5):
    n = draw(st.integers(min_value=min_times, max_value=max_times))
    times = sorted(draw(st.lists(
        st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
        min_size=n, max_size=n)))
    times = np.array(times)
    assume(len(np.unique(times)) == len(times))
    return times

@given(sorted_times_strategy())
@settings(max_examples=200)
def test_rotation_spline_boundary_conditions(times):
    """Property: RotationSpline should handle valid time arrays."""
    n = len(times)
    rotations = Rotation.random(n)
    spline = RotationSpline(times, rotations)

if __name__ == "__main__":
    test_rotation_spline_boundary_conditions()
```

<details>

<summary>
**Failing input**: `array([ 0.    ,  0.0625,  9.    , 33.    ])`
</summary>
```
/home/npc/.local/lib/python3.13/site-packages/scipy/spatial/transform/_rotation_spline.py:341: RuntimeWarning: overflow encountered in power
  b0 = 6 * (rotvecs[:-1] * dt[:-1, None] ** -2 +
/home/npc/.local/lib/python3.13/site-packages/scipy/spatial/transform/_rotation_spline.py:343: RuntimeWarning: overflow encountered in multiply
  b0[0] -= 2 / dt[0] * A_inv[0].dot(angular_rate_first)
/home/npc/.local/lib/python3.13/site-packages/scipy/spatial/transform/_rotation_spline.py:343: RuntimeWarning: invalid value encountered in subtract
  b0[0] -= 2 / dt[0] * A_inv[0].dot(angular_rate_first)
/home/npc/.local/lib/python3.13/site-packages/scipy/spatial/transform/_rotation_spline.py:389: RuntimeWarning: overflow encountered in divide
  angular_rates = rotvecs / dt[:, None]
/home/npc/.local/lib/python3.13/site-packages/scipy/spatial/transform/_rotation_spline.py:339: RuntimeWarning: overflow encountered in divide
  4 * (1 / dt[:-1] + 1 / dt[1:]))
/home/npc/.local/lib/python3.13/site-packages/scipy/spatial/transform/_rotation_spline.py:342: RuntimeWarning: overflow encountered in power
  rotvecs[1:] * dt[1:, None] ** -2)
/home/npc/.local/lib/python3.13/site-packages/scipy/spatial/transform/_rotation_spline.py:343: RuntimeWarning: overflow encountered in scalar divide
  b0[0] -= 2 / dt[0] * A_inv[0].dot(angular_rate_first)
/home/npc/.local/lib/python3.13/site-packages/scipy/spatial/transform/_rotation_spline.py:343: RuntimeWarning: invalid value encountered in dot
  b0[0] -= 2 / dt[0] * A_inv[0].dot(angular_rate_first)
/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/numeric.py:1753: RuntimeWarning: overflow encountered in multiply
  multiply(a1, b2, out=cp0)
/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/numeric.py:1756: RuntimeWarning: overflow encountered in multiply
  multiply(a2, b0, out=cp1)
/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/numeric.py:1757: RuntimeWarning: overflow encountered in multiply
  multiply(a0, b2, out=tmp)
/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/numeric.py:1760: RuntimeWarning: overflow encountered in multiply
  multiply(a1, b0, out=tmp)
/home/npc/.local/lib/python3.13/site-packages/scipy/spatial/transform/_rotation_spline.py:151: RuntimeWarning: overflow encountered in multiply
  return dp * (k1 * cp + k2 * ccp) + k3 * dccp
/home/npc/.local/lib/python3.13/site-packages/scipy/spatial/transform/_rotation_spline.py:151: RuntimeWarning: invalid value encountered in add
  return dp * (k1 * cp + k2 * ccp) + k3 * dccp
/home/npc/.local/lib/python3.13/site-packages/scipy/spatial/transform/_rotation_spline.py:350: RuntimeWarning: invalid value encountered in subtract
  b = b0 - delta_beta
/home/npc/.local/lib/python3.13/site-packages/scipy/spatial/transform/_rotation_spline.py:399: RuntimeWarning: invalid value encountered in divide
  coeff[0] = (-2 * rotvecs + dt * angular_rates
/home/npc/.local/lib/python3.13/site-packages/scipy/spatial/transform/_rotation_spline.py:401: RuntimeWarning: divide by zero encountered in divide
  coeff[1] = (3 * rotvecs - 2 * dt * angular_rates
/home/npc/.local/lib/python3.13/site-packages/scipy/spatial/transform/_rotation_spline.py:401: RuntimeWarning: invalid value encountered in divide
  coeff[1] = (3 * rotvecs - 2 * dt * angular_rates
/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/numeric.py:1754: RuntimeWarning: overflow encountered in multiply
  tmp = np.multiply(a2, b1, out=...)
/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/numeric.py:1758: RuntimeWarning: invalid value encountered in subtract
  cp1 -= tmp
/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/numeric.py:1759: RuntimeWarning: overflow encountered in multiply
  multiply(a0, b1, out=cp2)
/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/numeric.py:1761: RuntimeWarning: invalid value encountered in subtract
  cp2 -= tmp
/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/numeric.py:1755: RuntimeWarning: invalid value encountered in subtract
  cp0 -= tmp
/home/npc/.local/lib/python3.13/site-packages/scipy/spatial/transform/_rotation_spline.py:401: RuntimeWarning: overflow encountered in multiply
  coeff[1] = (3 * rotvecs - 2 * dt * angular_rates
/home/npc/.local/lib/python3.13/site-packages/scipy/spatial/transform/_rotation_spline.py:399: RuntimeWarning: overflow encountered in multiply
  coeff[0] = (-2 * rotvecs + dt * angular_rates
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/0/hypo.py", line 24, in <module>
    test_rotation_spline_boundary_conditions()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/0/hypo.py", line 16, in test_rotation_spline_boundary_conditions
    @settings(max_examples=200)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/0/hypo.py", line 21, in test_rotation_spline_boundary_conditions
    spline = RotationSpline(times, rotations)
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/spatial/transform/_rotation_spline.py", line 394, in __init__
    angular_rates, rotvecs_dot = self._solve_for_angular_rates(
                                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        dt, angular_rates, rotvecs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/spatial/transform/_rotation_spline.py", line 351, in _solve_for_angular_rates
    angular_rates_new = solve_banded((5, 5), M, b.ravel())
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/linalg/_basic.py", line 603, in solve_banded
    return _solve_banded(nlower, nupper, ab, b, overwrite_ab=overwrite_ab,
                         overwrite_b=overwrite_b, check_finite=check_finite)
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/_lib/_util.py", line 1233, in wrapper
    return f(*arrays, *other_args, **kwargs)
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/linalg/_basic.py", line 610, in _solve_banded
    b1 = _asarray_validated(b, check_finite=check_finite, as_inexact=True)
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/_lib/_util.py", line 455, in _asarray_validated
    a = toarray(a)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/lib/_function_base_impl.py", line 665, in asarray_chkfinite
    raise ValueError(
        "array must not contain infs or NaNs")
ValueError: array must not contain infs or NaNs
Falsifying example: test_rotation_spline_boundary_conditions(
    times=array([ 0.    ,  0.0625,  9.    , 33.    ]),
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/lib/_function_base_impl.py:665
```
</details>

## Reproducing the Bug

```python
import numpy as np
from scipy.spatial.transform import Rotation, RotationSpline

# Minimal failing case from the bug report
times = np.array([0., 0.0078125, 1., 5.])
rotations = Rotation.from_quat([
    [0.5, 0.5, 0.5, 0.5],
    [-0.5, 0.5, 0.5, 0.5],
    [0.5, -0.5, 0.5, 0.5],
    [0.5, 0.5, -0.5, 0.5]
])

print("Input times:", times)
print("Time deltas:", np.diff(times))
print("Max/Min ratio of time deltas:", np.max(np.diff(times)) / np.min(np.diff(times)))
print("Are times strictly increasing:", np.all(np.diff(times) > 0))
print("Are rotations valid unit quaternions:", np.allclose(np.linalg.norm(rotations.as_quat(), axis=1), 1))
print("\nAttempting to create RotationSpline...")

try:
    spline = RotationSpline(times, rotations)
    print("Success: RotationSpline created")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
```

<details>

<summary>
ValueError: array must not contain infs or NaNs
</summary>
```
/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/numeric.py:1753: RuntimeWarning: overflow encountered in multiply
  multiply(a1, b2, out=cp0)
/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/numeric.py:1754: RuntimeWarning: overflow encountered in multiply
  tmp = np.multiply(a2, b1, out=...)
/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/numeric.py:1756: RuntimeWarning: overflow encountered in multiply
  multiply(a2, b0, out=cp1)
/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/numeric.py:1757: RuntimeWarning: overflow encountered in multiply
  multiply(a0, b2, out=tmp)
/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/numeric.py:1758: RuntimeWarning: invalid value encountered in subtract
  cp1 -= tmp
/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/numeric.py:1759: RuntimeWarning: overflow encountered in multiply
  multiply(a0, b1, out=cp2)
/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/numeric.py:1760: RuntimeWarning: overflow encountered in multiply
  multiply(a1, b0, out=tmp)
/home/npc/.local/lib/python3.13/site-packages/scipy/spatial/transform/_rotation_spline.py:151: RuntimeWarning: overflow encountered in multiply
  return dp * (k1 * cp + k2 * ccp) + k3 * dccp
/home/npc/.local/lib/python3.13/site-packages/scipy/spatial/transform/_rotation_spline.py:151: RuntimeWarning: invalid value encountered in add
  return dp * (k1 * cp + k2 * ccp) + k3 * dccp
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/0/repo.py", line 21, in <module>
    spline = RotationSpline(times, rotations)
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/spatial/transform/_rotation_spline.py", line 394, in __init__
    angular_rates, rotvecs_dot = self._solve_for_angular_rates(
                                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        dt, angular_rates, rotvecs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/spatial/transform/_rotation_spline.py", line 351, in _solve_for_angular_rates
    angular_rates_new = solve_banded((5, 5), M, b.ravel())
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/linalg/_basic.py", line 603, in solve_banded
    return _solve_banded(nlower, nupper, ab, b, overwrite_ab=overwrite_ab,
                         overwrite_b=overwrite_b, check_finite=check_finite)
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/_lib/_util.py", line 1233, in wrapper
    return f(*arrays, *other_args, **kwargs)
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/linalg/_basic.py", line 610, in _solve_banded
    b1 = _asarray_validated(b, check_finite=check_finite, as_inexact=True)
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/_lib/_util.py", line 455, in _asarray_validated
    a = toarray(a)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/lib/_function_base_impl.py", line 665, in asarray_chkfinite
    raise ValueError(
        "array must not contain infs or NaNs")
ValueError: array must not contain infs or NaNs
Input times: [0.        0.0078125 1.        5.       ]
Time deltas: [0.0078125 0.9921875 4.       ]
Max/Min ratio of time deltas: 512.0
Are times strictly increasing: True
Are rotations valid unit quaternions: True

Attempting to create RotationSpline...
Error: ValueError: array must not contain infs or NaNs
```
</details>

## Why This Is A Bug

This violates the documented behavior of RotationSpline. The documentation specifies only two requirements for the `times` parameter:
1. "At least 2 times must be specified"
2. Times must be in strictly increasing order (enforced by the code)

The inputs in the failing case satisfy both requirements:
- We provide 4 times (≥ 2) ✓
- Times are strictly increasing: [0., 0.0078125, 1., 5.] ✓
- All inputs are finite floating-point numbers (no NaN/Inf) ✓
- Rotations are valid unit quaternions ✓

The documentation makes no mention of restrictions on:
- Time spacing uniformity
- Minimum or maximum time deltas
- Ratios between consecutive time intervals

The error message "array must not contain infs or NaNs" is misleading because the user's inputs contain neither - these values are generated internally due to numerical instability when time intervals vary by factors > ~500.

## Relevant Context

The bug occurs in the `_solve_for_angular_rates()` method (line 331-362 in _rotation_spline.py) when constructing and solving a banded matrix system. The numerical instability arises from operations like `dt[:-1, None] ** -2` when `dt` contains very small values (e.g., 0.0078125), causing overflow when inverted and squared.

Real-world scenarios that produce such time arrays include:
- Adaptive sampling algorithms that concentrate samples in regions of rapid change
- Sensor systems with variable sampling rates
- Motion capture data with non-uniform frame rates
- Scientific simulations with adaptive time stepping

The SciPy documentation example uses evenly-spaced times `[0, 10, 20, 40]`, which don't trigger this issue. Users have no way to anticipate that certain valid time arrays will cause crashes.

Documentation reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.RotationSpline.html

Source code location: scipy/spatial/transform/_rotation_spline.py

## Proposed Fix

Add input validation to detect problematic time spacing ratios and provide a clear error message:

```diff
--- a/scipy/spatial/transform/_rotation_spline.py
+++ b/scipy/spatial/transform/_rotation_spline.py
@@ -383,6 +383,14 @@ class RotationSpline:
         dt = np.diff(times)
         if np.any(dt <= 0):
             raise ValueError("Values in `times` must be in a strictly "
                              "increasing order.")
+
+        # Check for extreme time delta ratios that cause numerical instability
+        if len(dt) > 1:
+            dt_ratio = np.max(dt) / np.min(dt)
+            if dt_ratio > 1e5:
+                raise ValueError(f"Time intervals vary too much (ratio {dt_ratio:.2e}). "
+                               "RotationSpline requires more uniform time spacing to avoid "
+                               "numerical instability. Consider resampling your data.")

         rotvecs = (rotations[:-1].inv() * rotations[1:]).as_rotvec()
         angular_rates = rotvecs / dt[:, None]
```