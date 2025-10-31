# Bug Report: scipy.sparse.csgraph.csgraph_from_dense Silent Data Loss

**Target**: `scipy.sparse.csgraph.csgraph_from_dense` / `scipy.sparse.csgraph.csgraph_to_dense`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `csgraph_from_dense` function silently drops non-zero floating-point values smaller than approximately `1e-8`, violating the documented round-trip property and causing silent data corruption.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
import scipy.sparse.csgraph as csg

@settings(max_examples=500)
@given(st.integers(min_value=2, max_value=10).flatmap(
    lambda n: st.tuples(
        st.just(n),
        st.lists(st.lists(st.floats(min_value=0, max_value=100,
                                    allow_nan=False, allow_infinity=False),
                         min_size=n, max_size=n),
                min_size=n, max_size=n)
    )
))
def test_csgraph_round_trip_dense(args):
    n, graph_list = args
    graph = np.array(graph_list, dtype=float)

    csgraph_sparse = csg.csgraph_from_dense(graph, null_value=0)
    graph_reconstructed = csg.csgraph_to_dense(csgraph_sparse, null_value=0)

    assert graph_reconstructed.shape == graph.shape
    np.testing.assert_array_equal(graph, graph_reconstructed)
```

**Failing input**: `graph = [[0.0, 0.0], [0.0, 1.0018225238781444e-157]]`

## Reproducing the Bug

```python
import numpy as np
import scipy.sparse.csgraph as csg

graph = np.array([[0.0, 0.0], [0.0, 1.0018225238781444e-157]])

print(f"Original value at [1,1]: {graph[1,1]}")
print(f"Is exactly 0.0? {graph[1,1] == 0.0}")

sparse = csg.csgraph_from_dense(graph, null_value=0)
reconstructed = csg.csgraph_to_dense(sparse, null_value=0)

print(f"Sparse nnz: {sparse.nnz}")
print(f"Reconstructed value at [1,1]: {reconstructed[1,1]}")

assert graph[1,1] != 0.0
assert reconstructed[1,1] == 0.0
print("BUG: Non-zero value was silently dropped!")

for exp in [-10, -9, -8, -7]:
    val = 10.0 ** exp
    g = np.array([[0.0, val]])
    s = csg.csgraph_from_dense(g, null_value=0)
    print(f"Value {val:.2e}: nnz={s.nnz} ({'preserved' if s.nnz > 0 else 'DROPPED'})")
```

Output:
```
Original value at [1,1]: 1.0018225238781444e-157
Is exactly 0.0? False
Sparse nnz: 0
Reconstructed value at [1,1]: 0.0
BUG: Non-zero value was silently dropped!
Value 1.00e-10: nnz=0 (DROPPED)
Value 1.00e-09: nnz=0 (DROPPED)
Value 1.00e-08: nnz=1 (preserved)
Value 1.00e-07: nnz=1 (preserved)
```

## Why This Is A Bug

1. **Violates documented behavior**: The `null_value` parameter is documented as "Value that denotes non-edges in the graph. Default is zero." Users expect exact equality comparison with `null_value`, not an approximate comparison with an undocumented threshold.

2. **Silent data corruption**: Non-zero values that are mathematically distinct from zero are being silently dropped without any warning or error, leading to incorrect graph representations.

3. **Violates round-trip property**: `csgraph_to_dense(csgraph_from_dense(graph))` should equal `graph` for valid inputs, but this fails for small but non-zero values.

4. **Undocumented behavior**: There is no mention in the documentation of any threshold below which values are treated as null, nor any parameter to control this behavior.

5. **Arbitrary threshold**: The threshold of approximately `1e-8` appears arbitrary and has no clear mathematical or practical justification. Values like `1e-100` or `1e-50`, while small, are still valid floating-point numbers that may be meaningful in certain scientific applications.

## Fix

The bug is likely in the Cython implementation in `scipy/sparse/csgraph/_tools.pyx`. The comparison should use exact equality with `null_value` rather than an absolute tolerance-based comparison.

Expected behavior:
```python
# Current (buggy): uses some tolerance like abs(value) < 1e-8
if abs(value - null_value) < SOME_THRESHOLD:
    # treat as null

# Correct: use exact equality
if value == null_value:
    # treat as null
```

Alternatively, if tolerance-based comparison is intentional for numerical stability:
1. Document this threshold clearly in the docstring
2. Provide a parameter (e.g., `atol`) to allow users to control it
3. Use a comparison relative to `null_value`, not an absolute threshold: `abs(value - null_value) < atol`

Without access to the compiled Cython source, the exact line requiring the fix cannot be specified, but the issue is in the logic that determines which values are treated as null edges in `csgraph_from_dense` or `csgraph_masked_from_dense`.