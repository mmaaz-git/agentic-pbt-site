# Bug Report: scipy.spatial.Delaunay.find_simplex Fails to Locate Its Own Vertices

**Target**: `scipy.spatial.Delaunay.find_simplex`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `find_simplex()` method with default tolerance fails to locate points that are vertices of the Delaunay triangulation itself, incorrectly returning -1 (indicating "outside all simplices") for points that are definitionally part of the triangulation.

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

# Run the test with the specific failing case
if __name__ == "__main__":
    print("Running property-based test with Hypothesis...")
    print("Testing up to 500 random examples...")
    test_delaunay_point_location()
    print("\nAll random tests passed!")

    print("\nNow testing the specific failing case: n_points=13, seed=640")
    try:
        test_delaunay_point_location(13, 640)
        print("Test passed (unexpected!)")
    except AssertionError as e:
        print(f"Test failed as expected with error:\n{e}")
```

<details>

<summary>
**Failing input**: `n_points=13, seed=640`
</summary>
```
Running property-based test with Hypothesis...
Testing up to 500 random examples...
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/49/hypo.py", line 33, in <module>
    test_delaunay_point_location()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/49/hypo.py", line 7, in test_delaunay_point_location
    st.integers(min_value=10, max_value=30),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/49/hypo.py", line 21, in test_delaunay_point_location
    assert simplex_idx >= 0, \
           ^^^^^^^^^^^^^^^^
AssertionError: Input point 1 at [-1.10175507  1.79793862] not found in any simplex
Falsifying example: test_delaunay_point_location(
    n_points=13,
    seed=640,
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
from scipy.spatial import Delaunay

# Set the specific seed that causes the failure
np.random.seed(640)
points = np.random.randn(13, 2)

# Create Delaunay triangulation
tri = Delaunay(points)

# Get point 1 which should be part of the triangulation
point_1 = points[1]
print(f"Point 1 coordinates: {point_1}")
print()

# Try to find the simplex containing point 1 with default tolerance
result = tri.find_simplex(point_1)
print(f"find_simplex(point_1) with default tolerance: {result}")

# Check if point 1 is actually a vertex in the triangulation
print(f"\nPoint 1 is vertex of simplices: ", end="")
simplices_containing_point_1 = []
for s_idx, simplex in enumerate(tri.simplices):
    if 1 in simplex:
        simplices_containing_point_1.append(s_idx)
        print(s_idx, end=" ")
print()

# Verify that point 1 is indeed in the triangulation
print(f"\nVerification that point 1 is in triangulation:")
print(f"  Point 1 is in tri.points: {np.any(np.all(tri.points == point_1, axis=1))}")
print(f"  Point 1 index in tri.points: {np.where(np.all(tri.points == point_1, axis=1))[0]}")
print(f"  Is point 1 in coplanar list: {1 in tri.coplanar if len(tri.coplanar) > 0 else False}")

# Test with different tolerance values
print(f"\nTesting with different tolerance values:")
print(f"  With tol=None (default): {tri.find_simplex(point_1, tol=None)}")
print(f"  With tol=100*eps (2.22e-14): {tri.find_simplex(point_1, tol=100*np.finfo(float).eps)}")
print(f"  With tol=1e-14: {tri.find_simplex(point_1, tol=1e-14)}")
print(f"  With tol=1e-13: {tri.find_simplex(point_1, tol=1e-13)}")
print(f"  With tol=1e-12: {tri.find_simplex(point_1, tol=1e-12)}")
print(f"  With tol=1e-10: {tri.find_simplex(point_1, tol=1e-10)}")

# Additional check: show that this point is actually a vertex
print(f"\nConfirming point is a vertex of the triangulation:")
print(f"  tri.points[1] == point_1: {np.allclose(tri.points[1], point_1)}")
print(f"  Exact match: {np.array_equal(tri.points[1], point_1)}")
```

<details>

<summary>
find_simplex returns -1 for vertex that belongs to 4 simplices
</summary>
```
Point 1 coordinates: [-1.10175507  1.79793862]

find_simplex(point_1) with default tolerance: -1

Point 1 is vertex of simplices: 0 1 17 18

Verification that point 1 is in triangulation:
  Point 1 is in tri.points: True
  Point 1 index in tri.points: [1]
  Is point 1 in coplanar list: False

Testing with different tolerance values:
  With tol=None (default): -1
  With tol=100*eps (2.22e-14): -1
  With tol=1e-14: -1
  With tol=1e-13: 0
  With tol=1e-12: 0
  With tol=1e-10: 0

Confirming point is a vertex of the triangulation:
  tri.points[1] == point_1: True
  Exact match: True
```
</details>

## Why This Is A Bug

This violates the fundamental mathematical invariant that all vertices of a Delaunay triangulation must be locatable within the triangulation. The bug occurs because:

1. **Mathematical Incorrectness**: By definition, every vertex of a triangulation is part of at least one simplex. The function returning -1 (meaning "outside all simplices") for a vertex is mathematically incorrect.

2. **Silent Failure**: The function returns -1 without any warning or error, misleadingly indicating the point is outside the triangulation when it's actually a defining vertex of multiple simplices (0, 1, 17, and 18).

3. **Default Tolerance Insufficient**: The default tolerance (documented as `100*eps` ≈ 2.22e-14) is too small to handle the accumulated floating-point error when checking if vertices lie on simplex boundaries. The bug is fixed with tolerances ≥ 1e-13.

4. **Contradicts Documentation**: While the documentation doesn't explicitly guarantee vertices will be found, returning -1 contradicts the semantic meaning of "outside the triangulation" for points that define the triangulation itself.

5. **Breaks User Expectations**: Users reasonably expect that points used to construct a triangulation can be located within it, especially since the `vertex_to_simplex` attribute exists specifically for this purpose.

## Relevant Context

The issue stems from numerical precision when checking if points lie within simplices. Vertices lie exactly on simplex boundaries, where they are most susceptible to floating-point rounding errors. The Qhull library underlying SciPy's implementation uses geometric predicates that can accumulate small errors.

Key observations from testing:
- The point is confirmed to be vertex index 1 in `tri.points`
- The coordinates match exactly: `tri.points[1] == points[1]`
- The point is not in the coplanar list (wasn't excluded from triangulation)
- The tolerance threshold is precisely between 1e-14 (fails) and 1e-13 (works)

Related SciPy documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.find_simplex.html

The `find_simplex` method signature from the type stub shows:
```python
def find_simplex(
    self,
    xi: ArrayLike,
    bruteforce: bool = ...,
    tol: float = ...
) -> NDArray[np.intc]: ...
```

## Proposed Fix

Increase the default tolerance in `find_simplex` to handle numerical precision issues when locating vertices on simplex boundaries. The testing shows the minimum working tolerance is 1e-13, so 1e-10 provides a safe margin:

```diff
--- a/scipy/spatial/_qhull.pyx
+++ b/scipy/spatial/_qhull.pyx
@@ -2234,7 +2234,7 @@ class Delaunay(_QhullUser):
         tol : float, optional
             Tolerance allowed in the inside-triangle check.  Default is
-            ``100*eps``.
+            ``1e-10``.

         Returns
         -------
@@ -2256,7 +2256,7 @@ class Delaunay(_QhullUser):
         """
         cdef DelaunayInfo_t info
         cdef int isimplex
-        cdef double eps = 100 * np.finfo(np.double).eps
+        cdef double eps = 1e-10
         cdef double eps_broad = sqrt(eps)
         cdef int start
         cdef int k
```