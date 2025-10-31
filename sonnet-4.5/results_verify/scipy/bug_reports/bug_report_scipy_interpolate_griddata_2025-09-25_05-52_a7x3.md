# Bug Report: scipy.interpolate.griddata Returns NaN at Input Points

**Target**: `scipy.interpolate.griddata` and `scipy.interpolate.LinearNDInterpolator`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`griddata` with `method='linear'` or `method='cubic'` returns NaN for some input points when evaluating at those exact points, violating the fundamental interpolation property that an interpolator should return exact values at input points.

## Property-Based Test

```python
import numpy as np
import scipy.interpolate as interp
from hypothesis import given, strategies as st, assume, settings

@given(
    n=st.integers(min_value=3, max_value=20),
    seed=st.integers(min_value=0, max_value=10000)
)
@settings(max_examples=200)
def test_griddata_interpolates_at_input_points(n, seed):
    rng = np.random.default_rng(seed)

    points = rng.uniform(-10, 10, (n, 2))
    assume(len(np.unique(points, axis=0)) == n)

    values = rng.uniform(-10, 10, n)

    result = interp.griddata(points, values, points, method='linear')

    assert np.allclose(result, values, rtol=1e-10, atol=1e-10), \
        f"griddata should interpolate exact values at input points"
```

**Failing input**: `n=3, seed=6823`

## Reproducing the Bug

```python
import numpy as np
from scipy.interpolate import griddata

points = np.array([
    [5.57569186, 8.65211491],
    [0.9537184, -7.3914033],
    [5.84835361, 9.68439497]
])

values = np.array([1.30258009, 5.53827209, 9.65518382])

result = griddata(points, values, points, method='linear')

print("Expected:", values)
print("Got:     ", result)
```

Output:
```
Expected: [1.30258009 5.53827209 9.65518382]
Got:      [       nan 5.53827209        nan]
```

## Why This Is A Bug

Interpolation is defined as constructing a function that passes through a given set of points. A fundamental property of any interpolation method is that `interpolate(x_i) = y_i` for all input points `(x_i, y_i)`. This bug violates that property by returning NaN instead of the expected values.

The root cause is that `scipy.spatial.Delaunay.find_simplex` returns -1 when queried with points that are exactly at vertices of the triangulation, even though those points should be considered "inside" (or at least "on the boundary of") the simplices. When `LinearNDInterpolator` receives -1 from `find_simplex`, it treats the point as outside the triangulation and returns NaN.

## Fix

The bug is likely in `scipy.spatial.Delaunay.find_simplex` or in how `LinearNDInterpolator` handles vertex queries. A proper fix would need to:

1. Modify `find_simplex` to return a valid simplex index for points that are exactly at vertices, OR
2. Add special handling in `LinearNDInterpolator` to check if a query point matches any vertex before calling `find_simplex`, and return the vertex value directly if so

A simple workaround for users is to use `method='nearest'` instead of `method='linear'` or `method='cubic'`, as `NearestNDInterpolator` correctly handles vertex queries.