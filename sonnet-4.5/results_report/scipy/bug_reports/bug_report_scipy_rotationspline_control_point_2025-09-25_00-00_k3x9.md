# Bug Report: RotationSpline Fails to Pass Through Control Points

**Target**: `scipy.spatial.transform.RotationSpline`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

RotationSpline violates the fundamental interpolation property: it does not always pass through the specified control points. With certain non-uniform time spacings, the spline evaluates to a different rotation than the control rotation at that time.

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

def rotations_equal(r1, r2, atol=1e-5):
    q1 = r1.as_quat()
    q2 = r2.as_quat()
    return np.allclose(q1, q2, atol=atol) or np.allclose(q1, -q2, atol=atol)

@given(sorted_times_strategy())
@settings(max_examples=200)
def test_rotation_spline_boundary_conditions(times):
    """Property: RotationSpline should exactly match control points."""
    n = len(times)
    rotations = Rotation.random(n)
    spline = RotationSpline(times, rotations)

    for i, t in enumerate(times):
        result = spline([t])
        assert rotations_equal(result, rotations[i], atol=1e-5), \
            f"RotationSpline doesn't match control point {i} at t={t}"
```

**Failing input**: `times=array([0., 0.015625, 1., 2.])`

## Reproducing the Bug

```python
import numpy as np
from scipy.spatial.transform import Rotation, RotationSpline

times = np.array([0., 0.015625, 1., 2.])
np.random.seed(44)
rotations = Rotation.random(4)

spline = RotationSpline(times, rotations)

for i, t in enumerate(times):
    result = spline([t])
    expected_quat = rotations[i].as_quat()
    result_quat = result.as_quat()

    diff1 = np.linalg.norm(expected_quat - result_quat)
    diff2 = np.linalg.norm(expected_quat + result_quat)
    min_diff = min(diff1, diff2)

    if min_diff > 1e-5:
        print(f"MISMATCH at t={t} (control point {i})")
        print(f"  Expected: {expected_quat}")
        print(f"  Got:      {result_quat}")
        print(f"  Difference: {min_diff}")
```

## Why This Is A Bug

An interpolation spline must, by definition, pass through all of its control points. This is a fundamental mathematical property that scipy's own test suite explicitly verifies in `test_rotation_spline.py::test_spline_properties()`:

```python
# From scipy's tests
assert_allclose(spline(times).as_euler('xyz', degrees=True), angles)
```

The bug appears to be triggered by specific patterns of non-uniform time spacing, particularly when there's a very small time delta followed by larger ones. The API documentation does not restrict time spacing - it only requires times be strictly increasing.

## Fix

The issue likely stems from numerical instability in the spline coefficient computation when time deltas vary significantly. The spline construction should either:

1. Add input validation to reject problematic time spacings with a clear error message, or
2. Improve numerical stability in `_solve_for_angular_rates()` and coefficient computation to handle non-uniform spacing robustly.

A potential fix would involve normalizing the time intervals internally or using a more numerically stable solver for the banded matrix system in `_solve_for_angular_rates()`.