# Bug Report: scipy.interpolate.LinearNDInterpolator Returns NaN at Original Data Point

**Target**: `scipy.interpolate.LinearNDInterpolator`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

LinearNDInterpolator incorrectly returns NaN when evaluated at one of its own input data points, violating the fundamental property that an interpolator should return exact values at the data points used to construct it.

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

<details>

<summary>
**Failing input**: `n=4, seed=4580`
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/14
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_linearndinterpolator_roundtrip FAILED                      [100%]

=================================== FAILURES ===================================
_____________________ test_linearndinterpolator_roundtrip ______________________

    @given(
>       n=st.integers(min_value=4, max_value=20),
                   ^^^
        seed=st.integers(min_value=0, max_value=10000)
    )

hypo.py:7:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

n = 4, seed = 4580

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

>       np.testing.assert_allclose(result, values, rtol=1e-8, atol=1e-8)
E       AssertionError:
E       Not equal to tolerance rtol=1e-08, atol=1e-08
E
E       nan location mismatch:
E        ACTUAL: array([-87.753752,        nan, -83.387412,  85.639064])
E        DESIRED: array([-87.753752,  24.166249, -83.387412,  85.639064])
E       Falsifying example: test_linearndinterpolator_roundtrip(
E           n=4,
E           seed=4580,
E       )
E       Explanation:
E           These lines were always and only run by failing examples:
E               /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/arrayprint.py:1009
E               /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/arrayprint.py:1010
E               /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/getlimits.py:498
E               /home/npc/miniconda/lib/python3.13/site-packages/numpy/testing/_private/utils.py:771

hypo.py:29: AssertionError
=========================== short test summary info ============================
FAILED hypo.py::test_linearndinterpolator_roundtrip - AssertionError:
============================== 1 failed in 0.67s ===============================
```
</details>

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

<details>

<summary>
AssertionError: BUG: NaN at index [1]
</summary>
```
Expected: [1. 2. 3. 4.]
Got:      [ 1. nan  3.  4.]
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/14/repo.py", line 19, in <module>
    assert not np.any(np.isnan(result)), "BUG: NaN at index " + str(np.where(np.isnan(result))[0])
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: BUG: NaN at index [1]
```
</details>

## Why This Is A Bug

LinearNDInterpolator is documented as a piecewise linear interpolator that performs "linear barycentric interpolation" on triangulated data. By definition, linear interpolation must pass through the data points used to construct it. When evaluating the interpolator at the exact coordinates of an input point, it should return the corresponding input value, not NaN.

The root cause is a numerical precision issue in the underlying Delaunay triangulation's `find_simplex` method. Investigation shows that the triangulation's `find_simplex` incorrectly returns -1 (indicating "outside convex hull") for point index 1, even though this point was used to construct the triangulation itself. This causes LinearNDInterpolator to return its fill_value (default NaN) instead of the correct interpolated value.

This violates the fundamental mathematical contract of interpolation and can cause unexpected failures in scientific computing applications that depend on interpolation being exact at training points.

## Relevant Context

The bug stems from the interaction between LinearNDInterpolator and the underlying Delaunay triangulation from scipy.spatial. When debugging, we can see:

- The Delaunay triangulation correctly includes all 4 points in its simplices
- However, `tri.find_simplex(points)` returns `[1, -1, 0, 0]`, incorrectly marking point 1 as outside (-1)
- This happens despite point 1 being a vertex of the triangulation itself
- The issue appears to be related to numerical precision in the simplex containment test

SciPy's LinearNDInterpolator relies on Qhull for Delaunay triangulation. The numerical tolerance issues in geometric predicates can cause points very close to simplex boundaries to be misclassified.

Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LinearNDInterpolator.html

## Proposed Fix

The most robust fix would be to add special handling in LinearNDInterpolator for evaluation points that exactly match (within numerical tolerance) the input points, bypassing the Delaunay simplex search entirely:

```diff
--- a/scipy/interpolate/_interpnd.pyx
+++ b/scipy/interpolate/_interpnd.pyx
@@ -xxx,x +xxx,x @@ class LinearNDInterpolator:
     def _evaluate_double(self, xi):
         cdef double* values = <double*>self.values.data
         cdef double* out
         cdef double* points = <double*>self.points.data
         cdef int i, j, k
         cdef int ndim = self.ndim
         cdef int npoints = self.npoints
         cdef double eps = 1e-12

         # Allocate output
         out = <double*>malloc(sizeof(double) * xi.shape[0])

         for i in range(xi.shape[0]):
+            # Check if xi[i] exactly matches any input point
+            for j in range(npoints):
+                dist = 0.0
+                for k in range(ndim):
+                    d = xi[i, k] - points[j * ndim + k]
+                    dist += d * d
+                if dist < eps * eps:
+                    out[i] = values[j]
+                    continue
+
             # Original simplex-based evaluation
             isimplex = self.tri.find_simplex(xi[i])
             if isimplex == -1:
                 out[i] = self.fill_value
             else:
                 # ... existing barycentric interpolation code
```

This approach ensures that evaluation at training points always returns exact values, avoiding the numerical issues in `find_simplex` while maintaining backward compatibility for all other evaluation points.