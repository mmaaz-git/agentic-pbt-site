# Bug Report: scipy.spatial.Delaunay.find_simplex Incorrectly Reports Vertices as Outside Triangulation in Batch Queries

**Target**: `scipy.spatial.Delaunay.find_simplex`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `find_simplex()` method incorrectly returns -1 (indicating "outside triangulation") for some vertices when querying multiple points in batch mode with the default algorithm, even though these same vertices are correctly found when queried individually or with the bruteforce algorithm.

## Property-Based Test

```python
import numpy as np
from hypothesis import assume, given, settings, strategies as st, example
from scipy.spatial import Delaunay


@given(
    st.integers(min_value=4, max_value=25),
    st.integers(min_value=2, max_value=5)
)
@settings(max_examples=500)
@example(8, 4)  # The specific failing case
def test_delaunay_vertices_found_by_find_simplex(n_points, n_dims):
    np.random.seed(0)  # Fix seed for reproducibility
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


if __name__ == "__main__":
    # Run the test - will fail on the example(8, 4) case
    test_delaunay_vertices_found_by_find_simplex()
```

<details>

<summary>
**Failing input**: `n_points=8, n_dims=4`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/42/hypo.py", line 33, in <module>
    test_delaunay_vertices_found_by_find_simplex()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/42/hypo.py", line 7, in test_delaunay_vertices_found_by_find_simplex
    st.integers(min_value=4, max_value=25),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
    _raise_to_user(errors, state.settings, [], " in explicit examples")
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/42/hypo.py", line 28, in test_delaunay_vertices_found_by_find_simplex
    assert np.all(~failed_default), f"All vertices should be found by default algorithm, but {np.sum(failed_default)} were not"
           ~~~~~~^^^^^^^^^^^^^^^^^
AssertionError: All vertices should be found by default algorithm, but 2 were not
Falsifying explicit example: test_delaunay_vertices_found_by_find_simplex(
    n_points=8,
    n_dims=4,
)
```
</details>

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
    default_simplex = simplex_indices_default[i]
    bruteforce_simplex = simplex_indices_bruteforce[i]
    # Also check individual query
    individual_simplex = tri.find_simplex(points[i:i+1])[0]
    print(f"Point {i}: is_vertex={is_vertex}, default_simplex={default_simplex}, bruteforce_simplex={bruteforce_simplex}, individual_query={individual_simplex}")

print(f"\nCoplanar points: {tri.coplanar}")
print(f"Number of simplices: {len(tri.simplices)}")
print(f"All input points appear as vertices: {all(i in tri.simplices.flatten() for i in range(8))}")
```

<details>

<summary>
Points 6 and 7 incorrectly reported as outside triangulation in batch mode
</summary>
```
Default algorithm: [ 9  9  0  0 10 10 -1 -1]
Bruteforce algorithm: [0 6 0 0 2 0 1 0]

Points 6 and 7 return -1 with default but are found with bruteforce
Point 6: is_vertex=True, default_simplex=-1, bruteforce_simplex=1, individual_query=8
Point 7: is_vertex=True, default_simplex=-1, bruteforce_simplex=0, individual_query=0

Coplanar points: []
Number of simplices: 14
All input points appear as vertices: True
```
</details>

## Why This Is A Bug

This violates the fundamental expectation that vertices of a triangulation are always "inside" the triangulation. The bug manifests as an inconsistency in the batch processing code path:

1. **Points 6 and 7 are confirmed vertices** - They appear in `tri.simplices` and are part of the triangulation structure.

2. **Batch vs Individual Query Inconsistency** - The same point returns different results depending on how it's queried:
   - Point 6 in batch mode with default algorithm: returns -1 (outside)
   - Point 6 queried individually with default algorithm: returns simplex 8 (inside)
   - Point 6 in batch mode with bruteforce: returns simplex 1 (inside)

3. **Documentation states -1 means "outside the triangulation"** - Yet these are vertices that define the triangulation itself. A vertex cannot be outside its own triangulation.

4. **The bruteforce algorithm works correctly** - This proves the points are actually inside and findable, indicating the issue is specifically in the optimized batch processing algorithm.

## Relevant Context

The scipy.spatial.Delaunay documentation mentions that "Qhull does not guarantee that each input point appears as a vertex in the Delaunay triangulation," but in this case:
- All 8 input points ARE vertices (verified in the output)
- The `coplanar` array is empty (no points were excluded)
- Individual queries with the same default algorithm succeed

The issue appears to be related to numerical tolerance in the batch processing optimization. The default tolerance of `100*eps` may be insufficient for reliable vertex detection in higher dimensions when processing multiple points simultaneously.

Documentation reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html

## Proposed Fix

The issue is in the batch processing code path of the default (non-bruteforce) algorithm. A proper fix would require modifying the C/Cython implementation to either:

1. Use a more conservative tolerance for batch queries
2. Special-case vertex detection before the general simplex search
3. Align the batch and individual query code paths to use the same tolerance logic

As a workaround, users can:
- Use `bruteforce=True` for guaranteed correctness
- Query points individually instead of in batches
- Increase tolerance: `tri.find_simplex(points, tol=1e-10)`

Since the implementation is in compiled code, a patch cannot be provided here, but the fix would involve adjusting the tolerance or algorithm consistency in the `_qhull.pyx` implementation of `find_simplex`.