# Bug Report: RotationSpline Produces Zero Norm Quaternions During Evaluation

**Target**: `scipy.spatial.transform.RotationSpline`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

RotationSpline successfully initializes with certain valid inputs but crashes with "Found zero norm quaternions" when evaluating the spline at intermediate points. This is a silent failure mode where construction succeeds but usage fails.

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
```

**Failing input**: `times=array([0., 0.0078125, 1., 4.])`

## Reproducing the Bug

```python
import numpy as np
from scipy.spatial.transform import Rotation, RotationSpline

times = np.array([0., 0.0078125, 1., 4.])
np.random.seed(43)
rotations = Rotation.random(4)

spline = RotationSpline(times, rotations)
print("Spline created successfully")

t_mid = 0.5
result = spline([t_mid])
```

Output:
```
Spline created successfully
ValueError: Found zero norm quaternions in `quat`.
```

The error occurs in the `__call__` method when composing rotations: `self.rotations[index] * Rotation.from_rotvec(rotvecs)`.

## Why This Is A Bug

The spline initialization accepts the inputs without error, leading users to believe the spline is valid. However, the numerical coefficients computed during initialization lead to invalid rotation vectors during evaluation, ultimately producing zero-norm quaternions.

This is particularly problematic because:
1. Construction succeeds, giving no indication of problems
2. The crash happens during normal usage (evaluation)
3. The inputs are valid according to all documented requirements
4. Users have no way to predict or prevent this failure

Silent failures that manifest later are worse than immediate validation errors because they can occur deep in production code paths.

## Fix

The issue stems from numerical instability in the cubic polynomial coefficients computed during initialization. When evaluated, these coefficients can produce rotation vectors that map to degenerate quaternions.

Potential fixes:

1. **Validate during construction**: Check that computed coefficients will produce valid rotations
2. **Improve numerical stability**: Use better-conditioned computation for the polynomial coefficients
3. **Runtime validation**: Add safeguards in `__call__` to handle near-zero rotation vectors

```diff
--- a/scipy/spatial/transform/_rotation_spline.py
+++ b/scipy/spatial/transform/_rotation_spline.py
@@ -440,6 +440,11 @@ class RotationSpline:
                 delta_times = times - self.times[index - 1]

             rotvecs = _evaluate_cubic_polynomials(delta_times, self._coeff[:, index - 1])
+
+            # Safeguard against zero-norm rotation vectors
+            norms = np.linalg.norm(rotvecs, axis=1, keepdims=True)
+            rotvecs = np.where(norms > 1e-10, rotvecs, 0)
+
             result = self.rotations[index] * Rotation.from_rotvec(rotvecs)
```

Better yet, validate the coefficients during construction to fail fast with a clear error message.