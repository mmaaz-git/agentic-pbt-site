# Bug Report: scipy.spatial.transform.RotationSpline Incorrect Keyframe Value

**Target**: `scipy.spatial.transform.RotationSpline`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

RotationSpline returns the wrong rotation at the last keyframe time when keyframes are closely spaced near the end of the interval.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, settings, strategies as st
from scipy.spatial.transform import Rotation, RotationSpline


@given(
    st.integers(min_value=2, max_value=10),
    st.integers(min_value=1, max_value=1000)
)
@settings(max_examples=500)
def test_rotation_spline_keyframe_exact(n_keyframes, seed):
    np.random.seed(seed)

    times = np.sort(np.random.rand(n_keyframes)) * 10
    times[0] = 0.0

    quats = np.random.randn(n_keyframes, 4)
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    rotations = Rotation.from_quat(quats)

    spline = RotationSpline(times, rotations)

    for i, t in enumerate(times):
        interp_rot = spline(t)
        expected_quat = rotations[i].as_quat()
        actual_quat = interp_rot.as_quat()

        assert np.allclose(expected_quat, actual_quat, atol=1e-6) or np.allclose(expected_quat, -actual_quat, atol=1e-6), \
            f"RotationSpline at keyframe time {t} should return exact keyframe rotation"
```

**Failing inputs**: `n_keyframes=5, seed=493` and `n_keyframes=4, seed=4`

## Reproducing the Bug

```python
import numpy as np
from scipy.spatial.transform import Rotation, RotationSpline

times = np.array([0.0, 6.64446271, 9.24290286, 9.3721529, 9.53565684])
quats = np.array([
    [0.70966851, 0.62720382, 0.3207292, 0.0108974],
    [0.46483691, 0.04322698, 0.86833592, -0.16748374],
    [-0.63456588, -0.47115133, 0.61052646, -0.05099029],
    [0.04451323, -0.38889422, -0.59025565, -0.70595901],
    [-0.29974423, -0.77272482, -0.38528129, -0.40571921]
])

rotations = Rotation.from_quat(quats)
spline = RotationSpline(times, rotations)

expected = rotations[-1].as_quat()
actual = spline(times[-1]).as_quat()

print(f"Expected: {expected}")
print(f"Actual:   {actual}")
```

The spline returns quaternion `[0.04451323, -0.38889422, -0.59025565, -0.70595901]` (the 4th keyframe) instead of `[-0.29974423, -0.77272482, -0.38528129, -0.40571921]` (the 5th keyframe) at time 9.53565684.

## Why This Is A Bug

Any interpolation scheme must return the exact keyframe value when evaluated at a keyframe time. RotationSpline violates this fundamental interpolation property at the last keyframe when keyframes are closely spaced near the end. This appears to be an indexing or boundary condition error. Slerp, which performs simpler spherical linear interpolation, correctly handles this case.

## Fix

The fix likely involves correcting the interval search or boundary handling logic in RotationSpline when evaluating at the last keyframe time, particularly when the last interval is very short.