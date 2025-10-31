# Bug Report: floyd_warshall Silent Failure with Fortran-Contiguous Arrays

**Target**: `scipy.sparse.csgraph.floyd_warshall`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`floyd_warshall` with `overwrite=True` silently fails when given a Fortran-contiguous array, returning the unmodified input array instead of computing shortest paths. The function raises a `ValueError("ndarray is not C-contiguous")` internally, but this exception is ignored, leading to incorrect results being returned without any warning to the user.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
from scipy.sparse.csgraph import floyd_warshall

@given(st.integers(min_value=3, max_value=12))
def test_floyd_warshall_overwrite_consistency(n):
    graph = np.random.rand(n, n) * 10
    graph_c = np.ascontiguousarray(graph, dtype=np.float64)
    graph_f = np.asfortranarray(graph, dtype=np.float64)

    dist_c = floyd_warshall(graph_c, directed=True, overwrite=False)
    dist_f = floyd_warshall(graph_f, directed=True, overwrite=True)

    np.testing.assert_allclose(dist_c, dist_f, rtol=1e-10, atol=1e-10)
```

**Failing input**: Any Fortran-contiguous array with `n >= 2`

## Reproducing the Bug

```python
import numpy as np
from scipy.sparse.csgraph import floyd_warshall

np.random.seed(0)
graph = np.random.rand(3, 3) * 10
graph_f = np.asfortranarray(graph, dtype=np.float64)

print("Original graph (F-contiguous):")
print(graph_f)
print(f"Diagonal: {np.diag(graph_f)}")

result = floyd_warshall(graph_f, directed=True, overwrite=True)

print("\nResult after floyd_warshall:")
print(result)
print(f"Diagonal: {np.diag(result)}")
print(f"\nExpected diagonal: [0, 0, 0]")
print(f"Is result identical to input? {np.array_equal(result, graph_f)}")
```

Output:
```
Original graph (F-contiguous):
[[5.48813504 7.15189366 6.02763376]
 [5.44883183 4.23654799 6.45894113]
 [4.37587211 8.91773001 9.63662761]]
Diagonal: [5.48813504 4.23654799 9.63662761]

Result after floyd_warshall:
[[5.48813504 7.15189366 6.02763376]
 [5.44883183 4.23654799 6.45894113]
 [4.37587211 8.91773001 9.63662761]]
Diagonal: [5.48813504 4.23654799 9.63662761]

Expected diagonal: [0, 0, 0]
Is result identical to input? True
```

The function also prints to stderr (though this is suppressed in normal use):
```
ValueError: ndarray is not C-contiguous
Exception ignored in: 'scipy.sparse.csgraph._shortest_path._floyd_warshall'
```

## Why This Is A Bug

1. **Silently returns incorrect results**: The function returns the unmodified input instead of shortest paths, with no indication of failure. The diagonal should be all zeros for shortest paths (distance from each node to itself is 0), but instead contains the original values.

2. **Exception is ignored**: The `ValueError` is raised but then caught and ignored somewhere in the call stack, violating Python's principle that "errors should never pass silently."

3. **Inconsistent with documentation**: The documentation states that with `overwrite=True`, the function "overwrite[s] csgraph with the result" if "csgraph is a dense, c-ordered array with dtype=float64". However, when these conditions are not met, the function should either:
   - Raise an error to alert the user
   - Fall back to non-overwrite behavior
   - Convert the array to C-contiguous format

   Instead, it does none of these and returns garbage results.

4. **Data corruption risk**: Users who call this function with Fortran-contiguous arrays will receive incorrect results and may not notice, leading to downstream errors in their algorithms.

## Fix

The fix should ensure that when `overwrite=True` is requested but the array is not C-contiguous, the function either:

**Option 1**: Raise a clear error:
```python
if overwrite and not csgraph.flags['C_CONTIGUOUS']:
    raise ValueError("overwrite=True requires a C-contiguous array. "
                     "Use np.ascontiguousarray() or set overwrite=False.")
```

**Option 2**: Silently convert to C-contiguous:
```python
if overwrite and not csgraph.flags['C_CONTIGUOUS']:
    csgraph = np.ascontiguousarray(csgraph)
```

**Option 3**: Fall back to non-overwrite mode:
```python
if overwrite and not csgraph.flags['C_CONTIGUOUS']:
    overwrite = False
    warnings.warn("Array is not C-contiguous; overwrite=True ignored")
```

The current behavior of catching and ignoring the exception is unacceptable and should be fixed. Option 1 (raising a clear error) is preferred as it's most explicit and helps users understand the requirements.