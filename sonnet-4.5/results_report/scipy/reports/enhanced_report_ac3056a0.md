# Bug Report: scipy.sparse.csgraph.csgraph_from_dense Silently Drops Small Non-Zero Values Below 1e-8 Threshold

**Target**: `scipy.sparse.csgraph.csgraph_from_dense`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `csgraph_from_dense` function incorrectly drops non-zero values smaller than approximately 1e-8, treating them as if they were equal to the `null_value` parameter, causing silent data loss in sparse graph representations and incorrect results in graph algorithms.

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

if __name__ == "__main__":
    # Run the test
    test_direct_edge_not_longer_than_shortest_path()
```

<details>

<summary>
**Failing input**: `graph=array([[0.e+00, 1.e-09], [0.e+00, 0.e+00]])`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/16/hypo.py", line 32, in <module>
    test_direct_edge_not_longer_than_shortest_path()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/16/hypo.py", line 20, in test_direct_edge_not_longer_than_shortest_path
    @settings(max_examples=100)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/16/hypo.py", line 27, in test_direct_edge_not_longer_than_shortest_path
    assert dist_matrix[i, j] <= graph[i, j] + 1e-9, \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Shortest path from 0 to 1 should not be longer than direct edge
Falsifying example: test_direct_edge_not_longer_than_shortest_path(
    graph=array([[0.e+00, 1.e-09],
           [0.e+00, 0.e+00]]),
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/16/hypo.py:28
```
</details>

## Reproducing the Bug

```python
import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.csgraph import csgraph_from_dense, shortest_path

print("=== Demonstrating csgraph_from_dense bug with small values ===\n")

# Test 1: Show that small values are dropped
print("Test 1: Small values are incorrectly dropped by csgraph_from_dense")
print("-" * 60)

test_values = [1e-300, 1e-100, 1e-50, 1e-20, 1e-10, 1e-9, 2e-8, 1e-8, 1e-7, 1e-5, 1e-3]

for val in test_values:
    graph = np.array([[0.0, val], [0.0, 0.0]])

    # Compare csgraph_from_dense with standard csr_array
    sparse_scipy = csr_array(graph)
    sparse_csgraph = csgraph_from_dense(graph, null_value=0)

    print(f"Value: {val:.2e}")
    print(f"  scipy.sparse.csr_array nnz: {sparse_scipy.nnz}")
    print(f"  csgraph_from_dense nnz: {sparse_csgraph.nnz}")
    if sparse_scipy.nnz != sparse_csgraph.nnz:
        print(f"  *** BUG: Value dropped! ***")
    print()

print("\nTest 2: Impact on shortest path computation")
print("-" * 60)

# Use the failing example from the hypothesis test
graph_with_small_edge = np.array([[0.00000000e+00, 1.69552992e-69],
                                   [0.00000000e+00, 0.00000000e+00]])

print(f"Graph with edge weight {graph_with_small_edge[0,1]:.2e}:")
print(graph_with_small_edge)
print()

# Compute shortest paths
dist_matrix = shortest_path(graph_with_small_edge, directed=True)

print(f"Shortest path from node 0 to node 1: {dist_matrix[0,1]}")
print(f"Direct edge weight from 0 to 1: {graph_with_small_edge[0,1]}")
print()

if np.isinf(dist_matrix[0,1]):
    print("*** BUG CONFIRMED: Edge was treated as non-existent! ***")
    print("The shortest path algorithm returns infinity, indicating no path exists,")
    print("even though there is a direct edge with weight 1.69552992e-69")
else:
    print("Edge was correctly recognized")

print("\nTest 3: Finding the exact threshold")
print("-" * 60)

# Binary search to find the threshold more precisely
low, high = 1e-10, 1e-7
while high - low > 1e-12:
    mid = (low + high) / 2
    graph = np.array([[0.0, mid], [0.0, 0.0]])
    sparse = csgraph_from_dense(graph, null_value=0)
    if sparse.nnz == 0:
        low = mid
    else:
        high = mid

print(f"Threshold is approximately between {low:.3e} and {high:.3e}")
print(f"Values <= {low:.3e} are incorrectly dropped")
print(f"Values >= {high:.3e} are correctly kept")
```

<details>

<summary>
Output showing small values are incorrectly treated as non-edges
</summary>
```
=== Demonstrating csgraph_from_dense bug with small values ===

Test 1: Small values are incorrectly dropped by csgraph_from_dense
------------------------------------------------------------
Value: 1.00e-300
  scipy.sparse.csr_array nnz: 1
  csgraph_from_dense nnz: 0
  *** BUG: Value dropped! ***

Value: 1.00e-100
  scipy.sparse.csr_array nnz: 1
  csgraph_from_dense nnz: 0
  *** BUG: Value dropped! ***

Value: 1.00e-50
  scipy.sparse.csr_array nnz: 1
  csgraph_from_dense nnz: 0
  *** BUG: Value dropped! ***

Value: 1.00e-20
  scipy.sparse.csr_array nnz: 1
  csgraph_from_dense nnz: 0
  *** BUG: Value dropped! ***

Value: 1.00e-10
  scipy.sparse.csr_array nnz: 1
  csgraph_from_dense nnz: 0
  *** BUG: Value dropped! ***

Value: 1.00e-09
  scipy.sparse.csr_array nnz: 1
  csgraph_from_dense nnz: 0
  *** BUG: Value dropped! ***

Value: 2.00e-08
  scipy.sparse.csr_array nnz: 1
  csgraph_from_dense nnz: 1

Value: 1.00e-08
  scipy.sparse.csr_array nnz: 1
  csgraph_from_dense nnz: 0
  *** BUG: Value dropped! ***

Value: 1.00e-07
  scipy.sparse.csr_array nnz: 1
  csgraph_from_dense nnz: 1

Value: 1.00e-05
  scipy.sparse.csr_array nnz: 1
  csgraph_from_dense nnz: 1

Value: 1.00e-03
  scipy.sparse.csr_array nnz: 1
  csgraph_from_dense nnz: 1


Test 2: Impact on shortest path computation
------------------------------------------------------------
Graph with edge weight 1.70e-69:
[[0.00000000e+00 1.69552992e-69]
 [0.00000000e+00 0.00000000e+00]]

Shortest path from node 0 to node 1: inf
Direct edge weight from 0 to 1: 1.69552992e-69

*** BUG CONFIRMED: Edge was treated as non-existent! ***
The shortest path algorithm returns infinity, indicating no path exists,
even though there is a direct edge with weight 1.69552992e-69

Test 3: Finding the exact threshold
------------------------------------------------------------
Threshold is approximately between 1.000e-08 and 1.000e-08
Values <= 1.000e-08 are incorrectly dropped
Values >= 1.000e-08 are correctly kept
```
</details>

## Why This Is A Bug

This violates the documented behavior of `csgraph_from_dense` in multiple critical ways:

1. **Documentation Contract Violation**: The function documentation explicitly states that only values **equal to** the `null_value` parameter (default 0) should be treated as non-edges. The documentation makes no mention of any epsilon threshold or automatic dropping of "small" values. Values like 1e-300 are mathematically distinct from 0 and should be preserved.

2. **Inconsistent with scipy.sparse**: The standard `scipy.sparse.csr_array` constructor correctly preserves all non-zero values regardless of magnitude, including values as small as 1e-300. This inconsistency within the same library is unexpected and problematic.

3. **Silent Data Corruption**: The function silently drops values below approximately 1e-8 without any warning, error, or documentation. This leads to incorrect results in downstream computations like shortest path algorithms, where edges appear to not exist when they actually do.

4. **Breaks Valid Use Cases**: Many scientific and engineering applications legitimately use very small non-zero values:
   - Quantum mechanics calculations with extremely small probabilities
   - Normalized similarity scores that can be arbitrarily small
   - High-precision scientific computations
   - Graph weights representing probabilities in Markov chains
   - Machine learning applications with highly normalized features

5. **Arbitrary Threshold**: The threshold of approximately 1e-8 appears completely arbitrary and is not based on any documented floating-point precision considerations. This is neither machine epsilon (approximately 2.22e-16 for float64) nor any other standard numerical tolerance.

## Relevant Context

- **scipy version**: 1.16.2
- **Python version**: 3.13
- **Source code location**: https://github.com/scipy/scipy/blob/main/scipy/sparse/csgraph/_tools.pyx
- **Related functions**: The bug is in the Cython implementation, likely in `csgraph_masked_from_dense` or `csgraph_from_masked` functions
- **Impact**: All graph algorithms that depend on `csgraph_from_dense` (including `shortest_path`, `dijkstra`, `bellman_ford`, etc.) will produce incorrect results when graphs contain small edge weights

The bug appears to stem from an epsilon-based comparison in the Cython code, where values are being compared to `null_value` with a tolerance rather than exact equality. This is likely an optimization attempt that has gone wrong, as exact floating-point comparison is generally safe when checking against a specific sentinel value like 0.

## Proposed Fix

Based on analysis of the Cython implementation pattern, the bug likely exists in the comparison logic within `_tools.pyx`. The fix should replace any epsilon-based comparison with exact equality checking:

```diff
# In scipy/sparse/csgraph/_tools.pyx
# Current buggy implementation (approximate):
-   if abs(graph[i,j] - null_value) < 1e-8:
-       # treat as null edge
-       continue
+   # Fixed implementation:
+   if graph[i,j] == null_value or (nan_null and isnan(graph[i,j])) or (infinity_null and isinf(graph[i,j])):
+       # treat as null edge
+       continue
```

Alternatively, if floating-point comparison safety is a concern, the threshold should be:
1. Documented clearly in the function docstring
2. Made configurable via a parameter (e.g., `null_tolerance=0`)
3. Set to a more reasonable default like machine epsilon * max(abs(values)) if needed

The current undocumented behavior with a hard-coded 1e-8 threshold is unacceptable for a scientific computing library where numerical precision is critical.