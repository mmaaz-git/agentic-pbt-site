# Bug Report: scipy.sparse.csgraph.csgraph_from_dense Silent Data Loss for Small Non-Zero Values

**Target**: `scipy.sparse.csgraph.csgraph_from_dense`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `csgraph_from_dense` function silently drops non-zero floating-point values smaller than approximately 1e-8, violating documented behavior and causing data loss without warning.

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

if __name__ == "__main__":
    test_csgraph_round_trip_dense()
```

<details>

<summary>
**Failing input**: `args=(2, [[0.0, 0.0], [0.0, 1.401298464324817e-45]])`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/18/hypo.py", line 26, in <module>
    test_csgraph_round_trip_dense()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/18/hypo.py", line 6, in test_csgraph_round_trip_dense
    @given(st.integers(min_value=2, max_value=10).flatmap(
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/18/hypo.py", line 23, in test_csgraph_round_trip_dense
    np.testing.assert_array_equal(graph, graph_reconstructed)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/testing/_private/utils.py", line 1051, in assert_array_equal
    assert_array_compare(operator.__eq__, actual, desired, err_msg=err_msg,
    ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                         verbose=verbose, header='Arrays are not equal',
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                         strict=strict)
                         ^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/testing/_private/utils.py", line 916, in assert_array_compare
    raise AssertionError(msg)
AssertionError:
Arrays are not equal

Mismatched elements: 1 / 4 (25%)
Max absolute difference among violations: 1.40129846e-45
Max relative difference among violations: inf
 ACTUAL: array([[0.000000e+00, 0.000000e+00],
       [0.000000e+00, 1.401298e-45]])
 DESIRED: array([[0., 0.],
       [0., 0.]])
Falsifying example: test_csgraph_round_trip_dense(
    args=(2, [[0.0, 0.0], [0.0, 1.401298464324817e-45]]),
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
import scipy.sparse.csgraph as csg

# Create a graph with a very small but non-zero value
graph = np.array([[0.0, 0.0], [0.0, 1.0018225238781444e-157]])

print("="*60)
print("DEMONSTRATING BUG: Silent Data Loss in csgraph_from_dense")
print("="*60)
print()

print("Original graph:")
print(graph)
print()

print(f"Value at position [1,1]: {graph[1,1]}")
print(f"Is this value exactly 0.0? {graph[1,1] == 0.0}")
print(f"Scientific notation: {graph[1,1]:.2e}")
print()

# Convert to sparse and back
print("Converting to sparse representation...")
sparse = csg.csgraph_from_dense(graph, null_value=0)
print(f"Sparse matrix has {sparse.nnz} non-zero elements")
print()

print("Converting back to dense...")
reconstructed = csg.csgraph_to_dense(sparse, null_value=0)
print()

print("Reconstructed graph:")
print(reconstructed)
print()

print(f"Reconstructed value at [1,1]: {reconstructed[1,1]}")
print(f"Is reconstructed value 0.0? {reconstructed[1,1] == 0.0}")
print()

# Verify the bug
print("VERIFICATION:")
print(f"Original value was non-zero: {graph[1,1] != 0.0}")
print(f"Reconstructed value is zero: {reconstructed[1,1] == 0.0}")
print(f"Data was lost: {graph[1,1] != reconstructed[1,1]}")
print()

if graph[1,1] != 0.0 and reconstructed[1,1] == 0.0:
    print("*** BUG CONFIRMED: Non-zero value was silently dropped! ***")
    print()

# Test threshold values
print("="*60)
print("TESTING THRESHOLD BEHAVIOR")
print("="*60)
print()

test_values = [1e-10, 1e-9, 1e-8, 1.1e-8, 1.5e-8, 1e-7, 1e-6]

for val in test_values:
    g = np.array([[0.0, val], [val, 0.0]])
    s = csg.csgraph_from_dense(g, null_value=0)
    r = csg.csgraph_to_dense(s, null_value=0)
    preserved = s.nnz > 0
    status = "PRESERVED" if preserved else "DROPPED"
    print(f"Value {val:.2e}: nnz={s.nnz:2d}, Status={status:9s}, Reconstructed correctly: {np.allclose(g, r, rtol=0, atol=0)}")
```

<details>

<summary>
Bug demonstration showing silent data loss
</summary>
```
============================================================
DEMONSTRATING BUG: Silent Data Loss in csgraph_from_dense
============================================================

Original graph:
[[0.00000000e+000 0.00000000e+000]
 [0.00000000e+000 1.00182252e-157]]

Value at position [1,1]: 1.0018225238781444e-157
Is this value exactly 0.0? False
Scientific notation: 1.00e-157

Converting to sparse representation...
Sparse matrix has 0 non-zero elements

Converting back to dense...

Reconstructed graph:
[[0. 0.]
 [0. 0.]]

Reconstructed value at [1,1]: 0.0
Is reconstructed value 0.0? True

VERIFICATION:
Original value was non-zero: True
Reconstructed value is zero: True
Data was lost: True

*** BUG CONFIRMED: Non-zero value was silently dropped! ***

============================================================
TESTING THRESHOLD BEHAVIOR
============================================================

Value 1.00e-10: nnz= 0, Status=DROPPED  , Reconstructed correctly: False
Value 1.00e-09: nnz= 0, Status=DROPPED  , Reconstructed correctly: False
Value 1.00e-08: nnz= 0, Status=DROPPED  , Reconstructed correctly: False
Value 1.10e-08: nnz= 2, Status=PRESERVED, Reconstructed correctly: True
Value 1.50e-08: nnz= 2, Status=PRESERVED, Reconstructed correctly: True
Value 1.00e-07: nnz= 2, Status=PRESERVED, Reconstructed correctly: True
Value 1.00e-06: nnz= 2, Status=PRESERVED, Reconstructed correctly: True
```
</details>

## Why This Is A Bug

This behavior violates expected functionality and documented behavior in several critical ways:

1. **Documentation Violation**: The `null_value` parameter is documented as "Value that denotes non-edges in the graph" with no mention of any tolerance or threshold. The documentation clearly implies exact equality comparison. Users reasonably expect that only values exactly equal to `null_value` would be treated as non-edges.

2. **Silent Data Corruption**: Non-zero values that are mathematically distinct from zero are being silently dropped without any warning, error, or documentation. This is particularly dangerous in scientific computing where small values often have significant meaning (e.g., quantum mechanical calculations, chemical concentrations, small probabilities).

3. **Arbitrary Undocumented Threshold**: The threshold appears to be between 1.0e-8 and 1.1e-8, which is neither at machine epsilon (~2.2e-16 for float64) nor documented anywhere. This threshold is well within normal scientific computing ranges, not just edge cases with denormalized numbers.

4. **Broken Round-Trip Property**: The existence of both `csgraph_from_dense` and `csgraph_to_dense` strongly implies they should be inverse operations. Users expect that `csgraph_to_dense(csgraph_from_dense(graph))` should equal `graph` for valid inputs, but this fails for small non-zero values.

5. **Inconsistent API Design**: The function provides explicit boolean parameters `nan_null` and `infinity_null` to control how NaN and infinity values are treated, but provides no similar control for small values. If tolerance-based comparison was intentional, it should have an `atol` parameter.

6. **No User Control**: Unlike the handling of NaN and infinity values, users have no way to disable or control this threshold behavior, making it impossible to work with legitimate small values.

## Relevant Context

The bug affects the core graph conversion functionality in scipy's sparse graph module, which is widely used in network analysis, pathfinding algorithms, and scientific computing applications. The function is implemented in Cython in `scipy/sparse/csgraph/_tools.pyx`.

Documentation reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.csgraph_from_dense.html

The threshold behavior is particularly problematic because:
- Physical constants like the Planck constant (6.626e-34 J⋅s) when normalized could fall below this threshold
- Probabilities in machine learning models can be in the 1e-10 to 1e-8 range
- Chemical concentrations often use values like 1e-9 M (nanomolar)
- The threshold is 8 orders of magnitude larger than machine epsilon, meaning it's not a floating-point precision issue

## Proposed Fix

The bug requires modification to the Cython implementation in `scipy/sparse/csgraph/_tools.pyx`. The exact fix depends on whether the threshold behavior is intentional or not:

**Option 1: Remove the threshold (recommended)**
```diff
# In the Cython implementation, change from threshold-based comparison:
- if abs(value - null_value) < 1e-8:  # or similar threshold check
+ if value == null_value:  # exact equality
     # treat as null edge
```

**Option 2: If threshold is necessary for numerical stability, make it configurable:**
```diff
def csgraph_from_dense(graph, null_value=0, nan_null=True, infinity_null=True,
+                      atol=0):
    """
    ...
    Parameters
    ----------
    ...
+   atol : float, optional
+       Absolute tolerance for null value comparison. Values where
+       abs(value - null_value) < atol are treated as null edges.
+       Default is 0 (exact comparison).
    ...
    """

    # In implementation:
-   if abs(value - null_value) < 1e-8:
+   if abs(value - null_value) < atol:
        # treat as null edge
```

The first option is preferred as it matches the documented behavior and user expectations. If numerical stability concerns exist, they should be addressed through explicit, documented, and controllable parameters rather than hidden thresholds.