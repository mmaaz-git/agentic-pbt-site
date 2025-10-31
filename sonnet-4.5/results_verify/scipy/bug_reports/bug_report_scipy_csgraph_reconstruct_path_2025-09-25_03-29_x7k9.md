# Bug Report: scipy.sparse.csgraph.reconstruct_path Crash

**Target**: `scipy.sparse.csgraph.reconstruct_path`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`reconstruct_path` crashes with "ValueError: object too deep for desired array" when given valid predecessor matrices from `dijkstra`, even though the documentation states it should work with such input.

## Property-Based Test

```python
import numpy as np
import scipy.sparse.csgraph as csgraph
from hypothesis import given, strategies as st, settings


@given(st.integers(min_value=3, max_value=8))
@settings(max_examples=30)
def test_reconstruct_path_on_dijkstra_output(n):
    """
    Property: reconstruct_path should accept predecessor matrices from dijkstra

    This test creates a simple directed graph with some disconnected components,
    runs dijkstra to get distances and predecessors, then calls reconstruct_path
    with those predecessors. This should work without crashing.
    """
    G = np.zeros((n, n))

    for i in range(n//2):
        if i < n//2 - 1:
            G[i, i+1] = 1.0

    for i in range(n//2, n-1):
        G[i, i+1] = 1.0

    distances, predecessors = csgraph.dijkstra(G, directed=True, return_predecessors=True)

    paths = csgraph.reconstruct_path(G, predecessors, directed=True)

    assert paths.shape == distances.shape
```

**Failing input**: `n=3` (or any value >= 3)

## Reproducing the Bug

```python
import numpy as np
import scipy.sparse.csgraph as csgraph

G = np.array([[0., 0., 0.],
              [0., 0., 1.],
              [0., 0., 0.]])

distances, predecessors = csgraph.dijkstra(G, directed=True, return_predecessors=True)

paths = csgraph.reconstruct_path(G, predecessors, directed=True)
```

Output:
```
ValueError: object too deep for desired array
```

The graph has two disconnected components:
- Vertex 0 is isolated
- Vertices 1 and 2 are connected (1 -> 2)

The predecessor matrix from dijkstra is:
```
[[-9999 -9999 -9999]
 [-9999 -9999     1]
 [-9999 -9999 -9999]]
```

This is valid output from dijkstra, but reconstruct_path cannot handle it.

## Why This Is A Bug

The function `reconstruct_path` is documented to work with predecessor matrices. The docstring states it "Construct a tree from a graph and a predecessor matrix" and the predecessor matrix is exactly what dijkstra returns. The function should be able to handle graphs with disconnected components, as this is a common case.

## Fix

The bug appears to be in the Cython implementation at `scipy/sparse/csgraph/_tools.pyx:485`. The function likely has an issue with how it handles the nested arrays or object arrays when reconstructing paths for disconnected graphs.

A potential fix would be to:
1. Check if the implementation correctly handles cases where predecessors contain -9999 (indicating unreachable vertices)
2. Ensure the output array is initialized with the correct dtype and structure before population
3. Add proper handling for the case where entire rows/columns have no reachable paths

Without access to the source code, the exact fix cannot be specified, but the error message "object too deep for desired array" suggests an issue with array dimension or dtype handling in the Cython code.