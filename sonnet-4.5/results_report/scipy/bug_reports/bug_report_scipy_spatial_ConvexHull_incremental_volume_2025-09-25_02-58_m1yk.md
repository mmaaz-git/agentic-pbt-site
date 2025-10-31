# Bug Report: scipy.spatial.ConvexHull Incremental Volume Calculation

**Target**: `scipy.spatial.ConvexHull` (incremental mode)
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

ConvexHull with `incremental=True` computes incorrect volume after calling `add_points()`. The incremental hull reports a different (smaller) volume than an equivalent non-incremental hull built with all points at once, even when both hulls have identical vertices.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, assume, settings
from scipy.spatial import ConvexHull

@settings(max_examples=500)
@given(
    n_points=st.integers(min_value=3, max_value=20),
    n_dims=st.integers(min_value=2, max_value=3),
    data=st.data()
)
def test_convexhull_incremental_equals_batch(n_points, n_dims, data):
    points_list = data.draw(st.lists(
        st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100), min_size=n_dims, max_size=n_dims),
        min_size=n_points, max_size=n_points
    ))
    points = np.array(points_list)

    new_point_list = data.draw(st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100), min_size=n_dims, max_size=n_dims))
    new_point = np.array([new_point_list])

    try:
        hull_incremental = ConvexHull(points, incremental=True)
        hull_incremental.add_points(new_point)

        all_points = np.vstack([points, new_point])
        hull_batch = ConvexHull(all_points)

        assert np.isclose(hull_incremental.volume, hull_batch.volume), \
            f"Incremental and batch ConvexHull should have same volume: {hull_incremental.volume} vs {hull_batch.volume}"
    except Exception as e:
        if "QhullError" in str(type(e).__name__):
            assume(False)
        raise
```

**Failing input**:
- Initial points: `[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]`
- New point: `[0.0, 2.0]`

## Reproducing the Bug

```python
import numpy as np
from scipy.spatial import ConvexHull

points = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
new_point = np.array([[0.0, 2.0]])

hull_incremental = ConvexHull(points, incremental=True)
hull_incremental.add_points(new_point)

all_points = np.vstack([points, new_point])
hull_batch = ConvexHull(all_points)

print(f"Incremental volume: {hull_incremental.volume}")
print(f"Batch volume: {hull_batch.volume}")
```

Output:
```
Incremental volume: 0.6666666666666667
Batch volume: 1.0
```

## Why This Is A Bug

The incremental and batch ConvexHull constructions use the same final set of points and produce the same vertices, but report different volumes. The batch construction reports the correct volume (1.0, which can be verified by manual calculation for the triangle with vertices at `[0, 0]`, `[1, 0]`, and `[0, 2]`), while the incremental construction reports an incorrect volume (0.666...).

This violates the fundamental property that incremental construction should be equivalent to batch construction when given the same final set of points.

## Fix

The bug appears to be in the volume calculation after `add_points()` is called on an incremental ConvexHull. The volume should be recalculated correctly based on the updated hull geometry. Without access to the internal Qhull implementation details, the fix likely involves ensuring that volume/area attributes are properly updated when new points are added incrementally.

A potential workaround for users is to rebuild the hull non-incrementally when accurate volume is needed:
```python
hull_corrected = ConvexHull(hull_incremental.points)
correct_volume = hull_corrected.volume
```
