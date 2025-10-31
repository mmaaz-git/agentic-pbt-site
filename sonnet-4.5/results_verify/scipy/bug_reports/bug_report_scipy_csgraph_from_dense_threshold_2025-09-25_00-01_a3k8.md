# Bug Report: scipy.sparse.csgraph.csgraph_from_dense Incorrectly Drops Small Non-Zero Values

**Target**: `scipy.sparse.csgraph.csgraph_from_dense`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`csgraph_from_dense` incorrectly treats values smaller than approximately 1e-10 as if they equal the `null_value`, dropping them from the sparse representation even though they are mathematically distinct from zero.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings
from scipy.sparse.csgraph import shortest_path


@st.composite
def positive_weighted_graphs(draw, max_size=10):
    n = draw(st.integers(min_value=2, max_value=max_size))
    matrix = draw(st.lists(
        st.lists(st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
                 min_size=n, max_size=n),
        min_size=n, max_size=n
    ))
    matrix = np.array(matrix)
    np.fill_diagonal(matrix, 0)
    return matrix


@given(positive_weighted_graphs())
@settings(max_examples=100)
def test_direct_edge_not_longer_than_shortest_path(graph):
    dist_matrix = shortest_path(graph, directed=True)

    for i in range(graph.shape[0]):
        for j in range(graph.shape[1]):
            if graph[i, j] > 0:
                assert dist_matrix[i, j] <= graph[i, j] + 1e-9, \
                    f"Shortest path from {i} to {j} should not be longer than direct edge"
```

**Failing input**: `graph=array([[0.00000000e+00, 1.69552992e-69], [0.00000000e+00, 0.00000000e+00]])`

## Reproducing the Bug

```python
import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.csgraph import csgraph_from_dense

test_values = [1e-300, 1e-100, 1e-50, 1e-20, 1e-10, 1e-5, 1e-3]

for val in test_values:
    graph = np.array([[0.0, val], [0.0, 0.0]])

    sparse_scipy = csr_array(graph)
    sparse_csgraph = csgraph_from_dense(graph, null_value=0)

    print(f"Value: {val:.2e}")
    print(f"  scipy.sparse.csr_array nnz: {sparse_scipy.nnz}")
    print(f"  csgraph_from_dense nnz: {sparse_csgraph.nnz}")
```

Output:
```
Value: 1.00e-300
  scipy.sparse.csr_array nnz: 1
  csgraph_from_dense nnz: 0

Value: 1.00e-100
  scipy.sparse.csr_array nnz: 1
  csgraph_from_dense nnz: 0

Value: 1.00e-50
  scipy.sparse.csr_array nnz: 1
  csgraph_from_dense nnz: 0

Value: 1.00e-20
  scipy.sparse.csr_array nnz: 1
  csgraph_from_dense nnz: 0

Value: 1.00e-10
  scipy.sparse.csr_array nnz: 1
  csgraph_from_dense nnz: 0

Value: 1.00e-05
  scipy.sparse.csr_array nnz: 1
  csgraph_from_dense nnz: 1

Value: 1.00e-03
  scipy.sparse.csr_array nnz: 1
  csgraph_from_dense nnz: 1
```

## Why This Is A Bug

1. **Violates documented behavior**: The documentation states that only values equal to `null_value`, `inf`, or `NaN` should be treated as non-edges. There is no mention of any epsilon threshold.

2. **Inconsistent with scipy.sparse**: The standard `scipy.sparse.csr_array` constructor correctly preserves all non-zero values, even extremely small ones like 1e-300.

3. **Breaks legitimate use cases**: Users may have valid very small edge weights such as:
   - Probabilities in probabilistic graphs
   - Physical quantities with very small magnitudes
   - Normalized similarity scores

4. **Silent data loss**: The function silently drops these values without warning, leading to incorrect results in shortest path computations.

## Fix

The Cython implementation in `scipy/sparse/csgraph/_tools.pyx` likely uses an epsilon comparison like `abs(value - null_value) < epsilon` instead of exact equality `value == null_value`. The fix should use exact comparison:

```diff
-   if abs(value - null_value) < EPSILON:
+   if value == null_value:
        # treat as null edge
```

Or if NaN comparison is needed, use a proper equality check that handles all the documented special cases without introducing an arbitrary threshold.