# Bug Report: csgraph_from_dense Incorrect Null Value Comparison

**Target**: `scipy.sparse.csgraph.csgraph_from_dense`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`csgraph_from_dense` incorrectly treats values that are close to but not equal to `null_value` as null edges, using an inappropriate tolerance for floating-point comparison. This causes legitimate edges to be silently dropped when their weights are numerically close to the null value.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
from scipy.sparse.csgraph import csgraph_from_dense, csgraph_to_dense

@given(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False))
def test_csgraph_from_dense_null_value_handling(null_val):
    graph = np.array([
        [null_val, 1.0, 2.0],
        [null_val, null_val, 3.0],
        [null_val, null_val, null_val]
    ])

    sparse = csgraph_from_dense(graph, null_value=null_val)
    reconstructed = csgraph_to_dense(sparse, null_value=null_val)

    np.testing.assert_array_almost_equal(graph, reconstructed)
```

**Failing input**: `null_val=1.00001`

## Reproducing the Bug

```python
import numpy as np
from scipy.sparse.csgraph import csgraph_from_dense, csgraph_to_dense

null_val = 1.00001
graph = np.array([
    [null_val, 1.0, 2.0],
    [null_val, null_val, 3.0],
    [null_val, null_val, null_val]
])

print("Original graph:")
print(graph)

sparse = csgraph_from_dense(graph, null_value=null_val)
print(f"\nSparse representation: {sparse.nnz} non-zero elements")
print(f"Expected: 3 edges (1.0, 2.0, 3.0)")
print(f"Actual data: {sparse.data}")

reconstructed = csgraph_to_dense(sparse, null_value=null_val)
print("\nReconstructed graph:")
print(reconstructed)
print("\nDifference from original:")
print(graph - reconstructed)
```

Output:
```
Original graph:
[[1.00001 1.      2.     ]
 [1.00001 1.00001 3.     ]
 [1.00001 1.00001 1.00001]]

Sparse representation: 2 non-zero elements
Expected: 3 edges (1.0, 2.0, 3.0)
Actual data: [2. 3.]

Reconstructed graph:
[[1.00001 1.00001 2.     ]
 [1.00001 1.00001 3.     ]
 [1.00001 1.00001 1.00001]]

Difference from original:
[[ 0.e+00 -1.e-05  0.e+00]
 [ 0.e+00  0.e+00  0.e+00]
 [ 0.e+00  0.e+00  0.e+00]]
```

## Why This Is A Bug

The edge with weight `1.0` at position `[0, 1]` is incorrectly treated as a null edge because it is close to `null_value=1.00001`. The function uses an approximate comparison with too large a tolerance (appears to be around `1e-5` in relative terms), causing it to misclassify edges.

This violates the documented behavior: `null_value` specifies the exact value that denotes non-edges, not an approximate range. Users who legitimately want to use a graph with edges weighted `1.0` and null edges marked as `1.00001` (or vice versa) will silently lose data.

The bug affects multiple cases:
- `edge_val=1.0, null_val=1.00001`: Edge incorrectly dropped
- `edge_val=1.0, null_val=0.99999`: Edge incorrectly dropped
- `edge_val=5.0, null_val=5.00001`: Edge incorrectly dropped
- `edge_val=-1.0, null_val=-1.00001`: Edge incorrectly dropped

Interestingly, `edge_val=0.0, null_val=0.00001` works correctly, suggesting inconsistent comparison logic.

## Fix

The comparison for null values should use exact equality, not approximate equality. If approximate comparison is needed for some reason, the tolerance should be much smaller (e.g., machine epsilon relative to the values) and well-documented.

Looking at the implementation, the fix would likely involve replacing approximate comparisons like:
```python
if abs(value - null_value) < tolerance:  # Current (buggy)
```

with exact comparisons:
```python
if value == null_value:  # Correct
```

or using a much tighter tolerance:
```python
if abs(value - null_value) <= max(abs(value), abs(null_value)) * np.finfo(dtype).eps * 2:
```