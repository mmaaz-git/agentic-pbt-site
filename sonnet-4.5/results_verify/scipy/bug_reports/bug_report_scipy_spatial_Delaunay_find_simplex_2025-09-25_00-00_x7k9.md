# Bug Report: scipy.spatial.Delaunay.find_simplex Fails to Locate Vertices

**Target**: `scipy.spatial.Delaunay.find_simplex`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`Delaunay.find_simplex()` with default tolerance fails to locate points that are vertices of the triangulation, returning -1 instead of a valid simplex index.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, assume, settings
from scipy.spatial import Delaunay


@given(
    st.integers(min_value=10, max_value=30),
    st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=500)
def test_delaunay_point_location(n_points, seed):
    np.random.seed(seed)
    points = np.random.randn(n_points, 2)

    try:
        tri = Delaunay(points)

        for i in range(len(points)):
            simplex_idx = tri.find_simplex(points[i])

            assert simplex_idx >= 0, \
                f"Input point {i} at {points[i]} not found in any simplex"

    except Exception as e:
        if "degenerate" in str(e).lower():
            assume(False)
        raise
```

**Failing input**: `n_points=13, seed=640`

## Reproducing the Bug

```python
import numpy as np
from scipy.spatial import Delaunay

np.random.seed(640)
points = np.random.randn(13, 2)

tri = Delaunay(points)
point_1 = points[1]

result = tri.find_simplex(point_1)
print(f"find_simplex(point_1) = {result}")

print(f"Point 1 is vertex of simplices: ", end="")
for s_idx, simplex in enumerate(tri.simplices):
    if 1 in simplex:
        print(s_idx, end=" ")
print()

print(f"With tol=1e-12: {tri.find_simplex(point_1, tol=1e-12)}")
```

Output:
```
find_simplex(point_1) = -1
Point 1 is vertex of simplices: 0 1 17 18
With tol=1e-12: 0
```

## Why This Is A Bug

Point 1 at `[-1.10175507, 1.79793862]` is a vertex of the Delaunay triangulation, appearing in simplices 0, 1, 17, and 18. The function `find_simplex()` should locate this point in at least one of these simplices, but with default tolerance (`tol=None`) it returns -1, indicating the point is outside all simplices.

The default tolerance documented as `100*eps` (approximately `2.2e-14`) appears insufficient for numerical precision when checking if vertices lie within their own simplices. Using `tol=1e-12` or any larger tolerance correctly locates the point.

This violates the fundamental invariant that all input points used to construct a Delaunay triangulation must be locatable within the triangulation.

## Fix

Increase the default tolerance in `find_simplex` from `100*eps` to `1e-10` to account for accumulated numerical error when checking if points lie within simplices. This provides robustness while maintaining precision for typical point location queries.

```diff
- Default tolerance: 100*eps (≈ 2.2e-14)
+ Default tolerance: 1e-10
```

Alternatively, document this limitation and recommend users explicitly set `tol=1e-10` when querying points that may be vertices of the triangulation.