# Bug Report: scipy.spatial.transform.RotationSpline Numerical Overflow

**Target**: `scipy.spatial.transform.RotationSpline`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

RotationSpline crashes with "array must not contain infs or NaNs" when time intervals are very small and rotations between those intervals are non-trivial. The issue arises from numerical overflow when computing angular rates (rotation vector / time delta).

## Property-Based Test

```python
import numpy as np
from hypothesis import given, settings, strategies as st
from scipy.spatial.transform import Rotation, RotationSpline


@given(
    st.floats(min_value=1e-10, max_value=1e-7),
    st.floats(min_value=np.pi/2, max_value=np.pi)
)
@settings(max_examples=100)
def test_spline_with_small_time_differences(dt_small, angle):
    r1 = Rotation.from_quat([0, 0, 0, 1])
    r2 = Rotation.from_euler('z', angle, degrees=False)
    r3 = Rotation.from_euler('z', 2*angle, degrees=False)

    times = np.array([0.0, dt_small, 1.0])
    rotations = Rotation.concatenate([r1, r2, r3])

    spline = RotationSpline(times, rotations)
```

**Failing input**: `dt_small=5e-8, angle=π`

## Reproducing the Bug

```python
import numpy as np
from scipy.spatial.transform import Rotation, RotationSpline

rotations = Rotation.from_quat([
    [0.5, 0.5, -0.5, -0.5],
    [-0.5, 0.5, -0.5, 0.5],
    [0.5, 0.5, 0.5, 0.5],
    [0.5, 0.5, 0.5, 0.5],
])

times = np.array([0.0, 5e-8, 0.5, 1.0])

spline = RotationSpline(times, rotations)
```

This raises:
```
ValueError: array must not contain infs or NaNs
```

With warnings showing overflow in `_angular_acceleration_nonlinear_term`:
```
RuntimeWarning: overflow encountered in multiply
RuntimeWarning: invalid value encountered in add
```

## Why This Is A Bug

1. The inputs are mathematically valid - the times are strictly increasing and the rotations are valid unit quaternions
2. The docstring does not warn about minimum time intervals or require validation of angular rates
3. The overflow occurs deep in the internal calculation (_rotation_spline.py:151) when computing angular acceleration
4. Real-world applications (e.g., high-frequency sensor data with occasional duplicates or near-duplicates) could trigger this

The root cause: When `dt` is very small (e.g., 5e-8) and the rotation angle is large (e.g., π), the angular rate `rotvec / dt` becomes enormous (> 60 million rad/s), causing overflow when squared or multiplied in later calculations.

## Fix

The fix should add validation or numerical safeguards:

```diff
diff --git a/scipy/spatial/transform/_rotation_spline.py b/scipy/spatial/transform/_rotation_spline.py
index abc1234..def5678 100644
--- a/scipy/spatial/transform/_rotation_spline.py
+++ b/scipy/spatial/transform/_rotation_spline.py
@@ -382,6 +382,14 @@ class RotationSpline:
         dt = np.diff(times)
         if np.any(dt <= 0):
             raise ValueError("Values in `times` must be in a strictly "
                              "increasing order.")
+
+        # Check for time intervals that would cause numerical overflow
+        rotvecs = (rotations[:-1].inv() * rotations[1:]).as_rotvec()
+        angular_rates = np.linalg.norm(rotvecs, axis=1) / dt
+        MAX_ANGULAR_RATE = 1e6  # rad/s
+        if np.any(angular_rates > MAX_ANGULAR_RATE):
+            raise ValueError(f"Time intervals too small relative to rotation magnitude. "
+                           f"Maximum angular rate: {np.max(angular_rates):.2e} rad/s exceeds {MAX_ANGULAR_RATE:.2e} rad/s")

         rotvecs = (rotations[:-1].inv() * rotations[1:]).as_rotvec()
```

Alternatively, the implementation could use a more numerically stable algorithm or rescale internally to avoid overflow.