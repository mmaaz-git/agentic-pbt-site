# Bug Report: RotationSpline Crashes with NaN/Inf on Valid Inputs

**Target**: `scipy.spatial.transform.RotationSpline`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

RotationSpline crashes with "array must not contain infs or NaNs" when initialized with certain valid, non-uniformly spaced time arrays. The inputs satisfy all documented requirements (strictly increasing times, valid rotations) but cause numerical overflow during angular rate computation.

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
```

**Failing input**: `times=array([0., 0.0078125, 1., 5.])`

## Reproducing the Bug

```python
import numpy as np
from scipy.spatial.transform import Rotation, RotationSpline

times = np.array([0., 0.0078125, 1., 5.])
rotations = Rotation.from_quat([
    [0.5, 0.5, 0.5, 0.5],
    [-0.5, 0.5, 0.5, 0.5],
    [0.5, -0.5, 0.5, 0.5],
    [0.5, 0.5, -0.5, 0.5]
])

spline = RotationSpline(times, rotations)
```

Output:
```
ValueError: array must not contain infs or NaNs
```

Full traceback points to `_solve_for_angular_rates()` → `solve_banded()` → numerical overflow.

## Why This Is A Bug

The inputs are completely valid according to the API documentation:
- Times are strictly increasing ✓
- Rotations are valid unit quaternions ✓
- No documented restriction on time spacing ✓

Real-world applications may have non-uniform time sampling (e.g., adaptive sampling, sensor data with variable rates). The library should either:
1. Handle these inputs correctly, or
2. Document the restrictions and validate inputs with clear error messages

Currently, the cryptic error occurs deep in the numerical solver, making it hard for users to understand what went wrong.

## Fix

The numerical instability occurs in `_solve_for_angular_rates()` when time deltas vary significantly. Potential fixes:

1. **Input validation**: Add explicit checks for problematic time spacing ratios
2. **Numerical stability**: Normalize time intervals internally before solving
3. **Better conditioning**: Use scaled/conditioned matrices in the banded solver

```diff
--- a/scipy/spatial/transform/_rotation_spline.py
+++ b/scipy/spatial/transform/_rotation_spline.py
@@ -378,6 +378,13 @@ class RotationSpline:
         dt = np.diff(times)
         if np.any(dt <= 0):
             raise ValueError("Values in `times` must be in a strictly "
                             "increasing order.")
+
+        # Check for extreme time delta ratios that cause numerical instability
+        dt_ratio = np.max(dt) / np.min(dt)
+        if dt_ratio > 1e6:
+            raise ValueError(f"Time deltas vary too much (ratio {dt_ratio:.2e}). "
+                           "RotationSpline requires time intervals to be "
+                           "within 6 orders of magnitude of each other.")
```

Alternatively, normalize the time array internally and work in normalized coordinates to improve numerical stability.