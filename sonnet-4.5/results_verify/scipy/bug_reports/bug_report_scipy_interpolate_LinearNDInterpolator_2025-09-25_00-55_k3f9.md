# Bug Report: LinearNDInterpolator Returns NaN at Original Data Point

**Target**: `scipy.interpolate.LinearNDInterpolator`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`LinearNDInterpolator` incorrectly returns `NaN` when evaluated at one of its own input data points, violating the fundamental property that an interpolator should return exact values at the data points used to construct it.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings, assume
import scipy.interpolate


@given(
    n=st.integers(min_value=4, max_value=20),
    seed=st.integers(min_value=0, max_value=10000)
)
@settings(max_examples=500)
def test_linearndinterpolator_roundtrip(n, seed):
    """
    LinearNDInterpolator should return exact values at the original data points.
    Linear interpolation should pass through the data points.
    """
    rng = np.random.RandomState(seed)

    x = rng.uniform(-10, 10, n)
    y = rng.uniform(-10, 10, n)
    points = np.c_[x, y]

    assume(len(np.unique(points, axis=0)) == n)

    values = rng.uniform(-100, 100, n)

    interp = scipy.interpolate.LinearNDInterpolator(points, values)
    result = interp(points)

    np.testing.assert_allclose(result, values, rtol=1e-8, atol=1e-8)
```

**Failing input**: `n=4, seed=4580`

## Reproducing the Bug

```python
import numpy as np
import scipy.interpolate

points = np.array([
    [-0.48953057729796079, 9.11803150734971624],
    [ 3.71118356839197538, 7.25356777445734480],
    [ 1.02032417702106315, 5.85900990099145957],
    [-9.62742521563387577, 0.26520807215710640]
])

values = np.array([1.0, 2.0, 3.0, 4.0])

interp = scipy.interpolate.LinearNDInterpolator(points, values)
result = interp(points)

print("Expected: [1. 2. 3. 4.]")
print(f"Got:      {result}")

assert not np.any(np.isnan(result)), "BUG: NaN at index " + str(np.where(np.isnan(result))[0])
```

Output:
```
Expected: [1. 2. 3. 4.]
Got:      [ 1. nan  3.  4.]
AssertionError: BUG: NaN at index [1]
```

## Why This Is A Bug

`LinearNDInterpolator` is documented as a linear interpolator that should pass through its input data points. When evaluating the interpolator at the same points used to construct it, it should return the exact input values, not `NaN`.

The root cause appears to be a numerical precision issue in the underlying Delaunay triangulation's point location algorithm (`find_simplex`). The algorithm incorrectly classifies one of the input points as being outside the triangulation, returning simplex index -1, which causes `LinearNDInterpolator` to return `NaN` for that point.

This violates the fundamental interpolation property and makes the interpolator unreliable even for its own training data.

## Fix

The fix requires improving numerical robustness in one of two places:

1. **Delaunay.find_simplex**: Improve the point location algorithm to handle points that are on or very close to simplex boundaries more robustly, especially for points that were used to construct the triangulation.

2. **LinearNDInterpolator**: Add special handling for evaluation points that exactly match input points, bypassing the Delaunay simplex search and directly returning the corresponding value.

The second approach would be more robust and is a common optimization in interpolation libraries:

```diff
--- a/scipy/interpolate/_ndgriddata.py
+++ b/scipy/interpolate/_ndgriddata.py
@@ -xxx,x +xxx,x @@ class LinearNDInterpolator:
     def __call__(self, *args):
         xi = _ndim_coords_from_arrays(args, ndim=self.points.shape[1])
         xi = self._check_call_shape(xi)

+        # Fast path: check if any query points exactly match input points
+        result = np.full(xi.shape[0], np.nan)
+        for i, x in enumerate(xi):
+            distances = np.sum((self.points - x)**2, axis=1)
+            min_idx = np.argmin(distances)
+            if distances[min_idx] < 1e-14:  # Exact match within numerical precision
+                result[i] = self.values[min_idx]
+                continue
+            # ... existing simplex-based interpolation code
+
         # existing code continues
```

Note: This is a conceptual fix. The actual implementation would need to integrate properly with the existing code structure and handle edge cases.