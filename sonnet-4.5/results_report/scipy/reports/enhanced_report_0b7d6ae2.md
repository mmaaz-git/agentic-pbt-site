# Bug Report: scipy.spatial.transform.RotationSpline Produces Zero Norm Quaternions with Closely-Spaced Time Points

**Target**: `scipy.spatial.transform.RotationSpline`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

RotationSpline fails with valid inputs when time points are closely spaced (~0.015 apart), either crashing during construction with "array must not contain infs or NaNs" or during evaluation with "Found zero norm quaternions".

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
def test_rotation_spline_produces_valid_rotations(times):
    """Property: RotationSpline should produce valid rotations at any time."""
    n = len(times)
    rotations = Rotation.random(n)
    spline = RotationSpline(times, rotations)

    test_times = []
    for i in range(len(times) - 1):
        test_times.append((times[i] + times[i+1]) / 2)

    if test_times:
        results = spline(test_times)

if __name__ == "__main__":
    test_rotation_spline_produces_valid_rotations()
```

<details>

<summary>
**Failing input**: `times=array([0.      , 0.015625, 1.      , 4.      ])`
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
/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/numeric.py:1755: RuntimeWarning: invalid value encountered in subtract
  cp0 -= tmp
/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/numeric.py:1761: RuntimeWarning: invalid value encountered in subtract
  cp2 -= tmp
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/43/hypo.py", line 31, in <module>
  |     test_rotation_spline_produces_valid_rotations()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/43/hypo.py", line 16, in test_rotation_spline_produces_valid_rotations
  |     @settings(max_examples=200)
  |                    ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/43/hypo.py", line 21, in test_rotation_spline_produces_valid_rotations
    |     spline = RotationSpline(times, rotations)
    |   File "/home/npc/.local/lib/python3.13/site-packages/scipy/spatial/transform/_rotation_spline.py", line 394, in __init__
    |     angular_rates, rotvecs_dot = self._solve_for_angular_rates(
    |                                  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
    |         dt, angular_rates, rotvecs)
    |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/.local/lib/python3.13/site-packages/scipy/spatial/transform/_rotation_spline.py", line 351, in _solve_for_angular_rates
    |     angular_rates_new = solve_banded((5, 5), M, b.ravel())
    |   File "/home/npc/.local/lib/python3.13/site-packages/scipy/linalg/_basic.py", line 603, in solve_banded
    |     return _solve_banded(nlower, nupper, ab, b, overwrite_ab=overwrite_ab,
    |                          overwrite_b=overwrite_b, check_finite=check_finite)
    |   File "/home/npc/.local/lib/python3.13/site-packages/scipy/_lib/_util.py", line 1233, in wrapper
    |     return f(*arrays, *other_args, **kwargs)
    |   File "/home/npc/.local/lib/python3.13/site-packages/scipy/linalg/_basic.py", line 610, in _solve_banded
    |     b1 = _asarray_validated(b, check_finite=check_finite, as_inexact=True)
    |   File "/home/npc/.local/lib/python3.13/site-packages/scipy/_lib/_util.py", line 455, in _asarray_validated
    |     a = toarray(a)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/lib/_function_base_impl.py", line 665, in asarray_chkfinite
    |     raise ValueError(
    |         "array must not contain infs or NaNs")
    | ValueError: array must not contain infs or NaNs
    | Falsifying example: test_rotation_spline_produces_valid_rotations(
    |     times=array([0.      , 0.015625, 2.      , 9.      ]),
    | )
    | Explanation:
    |     These lines were always and only run by failing examples:
    |         /home/npc/miniconda/lib/python3.13/site-packages/numpy/lib/_function_base_impl.py:665
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/43/hypo.py", line 28, in test_rotation_spline_produces_valid_rotations
    |     results = spline(test_times)
    |   File "/home/npc/.local/lib/python3.13/site-packages/scipy/spatial/transform/_rotation_spline.py", line 445, in __call__
    |     result = self.rotations[index] * Rotation.from_rotvec(rotvecs)
    |              ~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    |   File "scipy/spatial/transform/_rotation.pyx", line 2692, in scipy.spatial.transform._rotation.Rotation.__mul__
    |   File "scipy/spatial/transform/_rotation.pyx", line 870, in scipy.spatial.transform._rotation.Rotation.__init__
    | ValueError: Found zero norm quaternions in `quat`.
    | Falsifying example: test_rotation_spline_produces_valid_rotations(
    |     times=array([0.      , 0.015625, 1.      , 4.      ]),
    | )
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
import numpy as np
from scipy.spatial.transform import Rotation, RotationSpline

# Use a case where construction succeeds but evaluation fails
times = np.array([1., 1.0078125, 2., 4.])
np.random.seed(43)
rotations = Rotation.random(4)

print(f"Times: {times}")
print(f"Number of rotations: {len(rotations)}")

try:
    spline = RotationSpline(times, rotations)
    print("Spline created successfully")

    # Try to evaluate at a middle point
    t_mid = 1.5
    print(f"Evaluating spline at t={t_mid}")
    result = spline([t_mid])
    print(f"Result: {result}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
```

<details>

<summary>
ValueError: Found zero norm quaternions during evaluation
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/43/repo.py", line 19, in <module>
    result = spline([t_mid])
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/spatial/transform/_rotation_spline.py", line 445, in __call__
    result = self.rotations[index] * Rotation.from_rotvec(rotvecs)
             ~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  File "scipy/spatial/transform/_rotation.pyx", line 2692, in scipy.spatial.transform._rotation.Rotation.__mul__
  File "scipy/spatial/transform/_rotation.pyx", line 870, in scipy.spatial.transform._rotation.Rotation.__init__
ValueError: Found zero norm quaternions in `quat`.
Times: [1.        1.0078125 2.        4.       ]
Number of rotations: 4
Spline created successfully
Evaluating spline at t=1.5
Error occurred: ValueError: Found zero norm quaternions in `quat`.
```
</details>

## Why This Is A Bug

This violates expected behavior because:

1. **Valid inputs fail**: The inputs satisfy all documented requirements - strictly increasing times with at least 2 values, matching number of rotations, no NaNs or infinities
2. **No documented limitations**: The documentation doesn't mention any minimum spacing requirements between time points or numerical stability considerations
3. **Silent failure mode**: In some cases, construction succeeds but evaluation fails later, making the error hard to predict and debug
4. **Reasonable use case fails**: Time spacing of 0.0078125 seconds (~7.8ms) or 0.015625 seconds (~15.6ms) is reasonable for robotics/animation at 60-120+ Hz

The root cause is numerical instability when computing angular rates and polynomial coefficients with very small time intervals (dt). The computation `angular_rates = rotvecs / dt[:, None]` in line 389 produces very large values when dt is small, leading to overflow in subsequent calculations.

## Relevant Context

The bug manifests in two ways depending on the exact time values:

1. **Immediate failure during construction**: Some inputs cause overflow during the iterative solver in `_solve_for_angular_rates()`, resulting in NaN/Inf values that trigger "array must not contain infs or NaNs" error
2. **Delayed failure during evaluation**: Other inputs allow construction but produce coefficients that lead to zero-norm rotation vectors during interpolation, causing "Found zero norm quaternions" error at line 445

Key code locations:
- Initial angular rate computation: `/home/npc/.local/lib/python3.13/site-packages/scipy/spatial/transform/_rotation_spline.py:389`
- Iterative solver that can overflow: `_rotation_spline.py:351`
- Evaluation that produces zero-norm quaternions: `_rotation_spline.py:445`

Documentation reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.RotationSpline.html

## Proposed Fix

```diff
--- a/scipy/spatial/transform/_rotation_spline.py
+++ b/scipy/spatial/transform/_rotation_spline.py
@@ -385,6 +385,12 @@ class RotationSpline:
             raise ValueError("Values in `times` must be in a strictly "
                              "increasing order.")

+        # Check for numerical stability with closely-spaced time points
+        min_dt = np.min(dt)
+        if min_dt < 1e-3:
+            raise ValueError(f"Time intervals must be at least 1e-3 for numerical stability. "
+                           f"Found minimum interval: {min_dt}")
+
         rotvecs = (rotations[:-1].inv() * rotations[1:]).as_rotvec()
         angular_rates = rotvecs / dt[:, None]
```

Alternative fix with safeguards during evaluation:

```diff
--- a/scipy/spatial/transform/_rotation_spline.py
+++ b/scipy/spatial/transform/_rotation_spline.py
@@ -442,6 +442,11 @@ class RotationSpline:
             n_segments = len(self.times) - 1
             index[index > n_segments - 1] = n_segments - 1
+
+            # Safeguard against near-zero rotation vectors
+            norms = np.linalg.norm(rotvecs, axis=1)
+            mask = norms < 1e-10
+            rotvecs[mask] = 0
+
             result = self.rotations[index] * Rotation.from_rotvec(rotvecs)
```