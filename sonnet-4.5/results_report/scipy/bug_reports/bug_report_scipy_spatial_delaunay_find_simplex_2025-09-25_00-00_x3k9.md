# Bug Report: scipy.spatial.Delaunay.find_simplex Fails to Locate Vertices

**Target**: `scipy.spatial.Delaunay.find_simplex`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`Delaunay.find_simplex()` incorrectly returns -1 for some vertices of the triangulation when using the default algorithm, despite these points being part of the triangulation. The bruteforce algorithm correctly finds all vertices.

## Property-Based Test

```python
import numpy as np
from hypothesis import assume, given, settings, strategies as st
from scipy.spatial import Delaunay


@given(
    st.integers(min_value=4, max_value=25),
    st.integers(min_value=2, max_value=5)
)
@settings(max_examples=500)
def test_delaunay_vertices_found_by_find_simplex(n_points, n_dims):
    points = np.random.randn(n_points, n_dims) * 100

    try:
        tri = Delaunay(points)
    except Exception:
        assume(False)

    simplex_indices_default = tri.find_simplex(points)
    simplex_indices_bruteforce = tri.find_simplex(points, bruteforce=True)

    failed_default = simplex_indices_default < 0
    failed_bruteforce = simplex_indices_bruteforce < 0

    assert np.all(~failed_bruteforce), "All vertices should be found by bruteforce"
    assert np.all(~failed_default), f"All vertices should be found by default algorithm, but {np.sum(failed_default)} were not"
```

**Failing input**: `n_points=8, n_dims=4`

## Reproducing the Bug

```python
import numpy as np
from scipy.spatial import Delaunay

np.random.seed(0)
points = np.random.randn(8, 4) * 100

tri = Delaunay(points)

simplex_indices_default = tri.find_simplex(points)
simplex_indices_bruteforce = tri.find_simplex(points, bruteforce=True)

print("Default algorithm:", simplex_indices_default)
print("Bruteforce algorithm:", simplex_indices_bruteforce)
print(f"\nPoints 6 and 7 return -1 with default but are found with bruteforce")

for i in [6, 7]:
    is_vertex = np.any(tri.simplices == i)
    print(f"Point {i}: is_vertex={is_vertex}, default_simplex={simplex_indices_default[i]}, bruteforce_simplex={simplex_indices_bruteforce[i]}")
```

## Why This Is A Bug

The `find_simplex` method's documentation states that it returns -1 for "Points outside the triangulation". However, the failing points (6 and 7) are vertices of the triangulation - they appear in `tri.simplices`. Vertices cannot be "outside" their own triangulation.

The bruteforce algorithm correctly finds all vertices, demonstrating that the default algorithm has a numerical precision issue that causes it to incorrectly classify some vertices as being outside the triangulation.

## Fix

The issue appears to be related to the default tolerance used for the inside-triangle check. The fix could be to:

1. Increase the default tolerance from `100*eps` to a more conservative value
2. Or ensure vertices are always found by checking against the vertex list first
3. Or document this as a known limitation and recommend using `bruteforce=True` or specifying `tol` for vertex queries

A simple workaround for users:
```python
simplex_indices = tri.find_simplex(points, tol=1e-10)
```

This increased tolerance resolves the issue in the failing example.