# Bug Report: scipy.spatial.ConvexHull Incorrect Volume After Incremental Point Addition

**Target**: `scipy.spatial.ConvexHull`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

ConvexHull with `incremental=True` calculates incorrect volume after calling `add_points()`, returning 33% less than the mathematically correct value despite having identical vertices to the batch-constructed hull.

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

# Run the test
if __name__ == "__main__":
    test_convexhull_incremental_equals_batch()
```

<details>

<summary>
**Failing input**: Initial points: `[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]`, New point: `[0.0, 2.0]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/29/hypo.py", line 37, in <module>
    test_convexhull_incremental_equals_batch()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/29/hypo.py", line 6, in test_convexhull_incremental_equals_batch
    @given(

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/29/hypo.py", line 28, in test_convexhull_incremental_equals_batch
    assert np.isclose(hull_incremental.volume, hull_batch.volume), \
           ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Incremental and batch ConvexHull should have same volume: 0.6666666666666667 vs 1.0
Falsifying example: test_convexhull_incremental_equals_batch(
    n_points=3,
    n_dims=2,
    data=data(...),
)
Draw 1: [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
Draw 2: [0.0, 2.0]
```
</details>

## Reproducing the Bug

```python
import numpy as np
from scipy.spatial import ConvexHull

# Initial points and new point from the bug report
points = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
new_point = np.array([[0.0, 2.0]])

# Incremental construction
hull_incremental = ConvexHull(points, incremental=True)
print(f"Initial hull volume: {hull_incremental.volume}")
print(f"Initial hull vertices: {points[hull_incremental.vertices]}")
hull_incremental.add_points(new_point)
print(f"Incremental volume after add_points: {hull_incremental.volume}")
print(f"Incremental vertices after add_points: {hull_incremental.points[hull_incremental.vertices]}")

# Batch construction with all points
all_points = np.vstack([points, new_point])
hull_batch = ConvexHull(all_points)
print(f"Batch volume: {hull_batch.volume}")
print(f"Batch vertices: {hull_batch.points[hull_batch.vertices]}")

# Verification
print(f"\nVerification:")
print(f"Incremental and batch volumes match: {np.isclose(hull_incremental.volume, hull_batch.volume)}")
print(f"Volume difference: {abs(hull_incremental.volume - hull_batch.volume)}")
print(f"Percentage error: {abs(hull_incremental.volume - hull_batch.volume) / hull_batch.volume * 100:.1f}%")

# Manual calculation for triangle with vertices [0,0], [1,0], [0,2]
# Area = 0.5 * base * height = 0.5 * 1 * 2 = 1.0
print(f"\nManual calculation for triangle [0,0], [1,0], [0,2]:")
print(f"Expected area: 1.0 (0.5 * base(1) * height(2))")
```

<details>

<summary>
Output showing 33% volume calculation error
</summary>
```
Initial hull volume: 0.5
Initial hull vertices: [[0. 0.]
 [1. 0.]
 [0. 1.]]
Incremental volume after add_points: 0.6666666666666667
Incremental vertices after add_points: [[0. 0.]
 [1. 0.]
 [0. 2.]]
Batch volume: 1.0
Batch vertices: [[0. 0.]
 [1. 0.]
 [0. 2.]]

Verification:
Incremental and batch volumes match: False
Volume difference: 0.33333333333333326
Percentage error: 33.3%

Manual calculation for triangle [0,0], [1,0], [0,2]:
Expected area: 1.0 (0.5 * base(1) * height(2))
```
</details>

## Why This Is A Bug

This violates the fundamental mathematical property that geometric objects with identical vertices should have identical volumes. The bug manifests specifically when:

1. **Both hulls identify the same vertices**: The incremental and batch constructions correctly identify vertices [0,0], [1,0], [0,2]
2. **The batch construction is mathematically correct**: Area = 0.5 × base × height = 0.5 × 1 × 2 = 1.0
3. **The incremental construction returns wrong volume**: 0.666... (2/3) instead of 1.0, a 33% error
4. **The error is not a floating-point issue**: The difference is exactly 1/3, suggesting a systematic calculation error

The ConvexHull.volume property is documented as returning "Volume of the convex hull" with no caveats about incremental mode producing different results. Users of scientific computing libraries rightfully expect mathematically correct results regardless of construction method.

## Relevant Context

- **SciPy Version**: 1.16.2
- **Initial hull state**: Before `add_points()`, the hull correctly shows volume 0.5 for the initial triangle
- **After add_points()**: The vertices update correctly but volume calculation fails
- **Impact**: Scientific computations relying on incremental hull volumes will silently produce incorrect results
- **Workaround available**: Reconstruct hull non-incrementally: `hull_corrected = ConvexHull(hull_incremental.points)`

Documentation references:
- [SciPy ConvexHull documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html)
- The incremental parameter is described as allowing "to add new points incrementally" with no warning about volume calculation differences

## Proposed Fix

The bug likely resides in the volume recalculation logic after `add_points()` in the incremental mode. Since the vertices are correctly updated but the volume is not, the issue appears to be that the volume attribute is not being properly recalculated based on the new hull geometry. A high-level fix would involve:

1. Ensure that after `add_points()` completes and updates the hull vertices/simplices, trigger a complete recalculation of the volume property
2. Use the same volume calculation method that the batch construction uses
3. Add regression tests comparing incremental and batch volume calculations

Without access to the internal Qhull binding implementation, a specific patch cannot be provided, but the fix should ensure that `hull.volume` always reflects the current geometric state of the hull, regardless of how it was constructed.