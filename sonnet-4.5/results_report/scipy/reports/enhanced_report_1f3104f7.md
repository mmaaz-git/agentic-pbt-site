# Bug Report: scipy.sparse.csgraph.csgraph_from_masked Crashes with Scalar Mask

**Target**: `scipy.sparse.csgraph.csgraph_from_masked`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `csgraph_from_masked` function crashes with an `AxisError` when given a masked array that has no masked values, because NumPy optimizes the mask to a scalar `False` instead of maintaining it as an array.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings
from scipy.sparse.csgraph import csgraph_from_masked, csgraph_to_masked


@st.composite
def graph_matrices(draw, max_size=20):
    n = draw(st.integers(min_value=2, max_value=max_size))
    matrix = draw(st.lists(
        st.lists(st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
                 min_size=n, max_size=n),
        min_size=n, max_size=n
    ))
    return np.array(matrix)


@given(graph_matrices())
@settings(max_examples=200)
def test_csgraph_masked_roundtrip(graph):
    masked_graph = np.ma.masked_equal(graph, 0)
    sparse_graph = csgraph_from_masked(masked_graph)
    reconstructed = csgraph_to_masked(sparse_graph)

    assert np.allclose(masked_graph.data, reconstructed.data, equal_nan=True)
    assert np.array_equal(masked_graph.mask, reconstructed.mask)


if __name__ == "__main__":
    test_csgraph_masked_roundtrip()
```

<details>

<summary>
**Failing input**: `graph=array([[1., 1.], [1., 1.]])`
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/13/hypo.py", line 29, in <module>
  |     test_csgraph_masked_roundtrip()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/13/hypo.py", line 18, in test_csgraph_masked_roundtrip
  |     @settings(max_examples=200)
  |                    ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/13/hypo.py", line 21, in test_csgraph_masked_roundtrip
    |     sparse_graph = csgraph_from_masked(masked_graph)
    |   File "scipy/sparse/csgraph/_tools.pyx", line 78, in scipy.sparse.csgraph._tools.csgraph_from_masked
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/_methods.py", line 51, in _sum
    |     return umr_sum(a, axis, dtype, out, keepdims, initial, where)
    | numpy.exceptions.AxisError: axis 1 is out of bounds for array of dimension 0
    | Falsifying example: test_csgraph_masked_roundtrip(
    |     graph=array([[1., 1.],
    |            [1., 1.]]),
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/13/hypo.py", line 24, in test_csgraph_masked_roundtrip
    |     assert np.allclose(masked_graph.data, reconstructed.data, equal_nan=True)
    |            ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    | AssertionError
    | Falsifying example: test_csgraph_masked_roundtrip(
    |     graph=array([[0., 0.],
    |            [0., 0.]]),
    | )
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
import numpy as np
from scipy.sparse.csgraph import csgraph_from_masked

# Create a simple 2x2 matrix with all non-zero values
graph = np.array([[1., 1.], [1., 1.]])

# Create a masked array where we mask values equal to 0
# Since there are no zeros, nothing gets masked
masked_graph = np.ma.masked_equal(graph, 0)

# Print information about the mask
print(f"Graph array:\n{graph}")
print(f"\nMasked graph data:\n{masked_graph.data}")
print(f"\nMask: {masked_graph.mask}")
print(f"Mask type: {type(masked_graph.mask)}")
print(f"Mask shape: {masked_graph.mask.shape if hasattr(masked_graph.mask, 'shape') else 'No shape attribute'}")
print(f"Mask ndim: {masked_graph.mask.ndim if hasattr(masked_graph.mask, 'ndim') else 'No ndim attribute'}")

# Try to convert to sparse graph - this should crash
print("\nAttempting to convert to sparse graph...")
try:
    sparse_graph = csgraph_from_masked(masked_graph)
    print("Success! Sparse graph created.")
    print(f"Sparse graph:\n{sparse_graph}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
```

<details>

<summary>
AxisError when converting masked array with no masked values
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/13/repo.py", line 22, in <module>
    sparse_graph = csgraph_from_masked(masked_graph)
  File "scipy/sparse/csgraph/_tools.pyx", line 78, in scipy.sparse.csgraph._tools.csgraph_from_masked
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/_methods.py", line 51, in _sum
    return umr_sum(a, axis, dtype, out, keepdims, initial, where)
numpy.exceptions.AxisError: axis 1 is out of bounds for array of dimension 0
Graph array:
[[1. 1.]
 [1. 1.]]

Masked graph data:
[[1. 1.]
 [1. 1.]]

Mask: False
Mask type: <class 'numpy.bool'>
Mask shape: ()
Mask ndim: 0

Attempting to convert to sparse graph...
Error occurred: AxisError: axis 1 is out of bounds for array of dimension 0
```
</details>

## Why This Is A Bug

This violates the documented behavior of `csgraph_from_masked` which states it accepts a `MaskedArray` as input without any qualification about the mask format. The function crashes when encountering a standard NumPy optimization where masks with no masked values are represented as a scalar `False` rather than an array of `False` values.

This is a documented and expected behavior of NumPy's MaskedArray implementation - when no values are masked, NumPy optimizes memory usage by storing the mask as a scalar `False` with shape `()` and ndim `0`. The `csgraph_from_masked` function attempts to perform operations on this mask assuming it's always a 2D array, specifically trying to sum along `axis=1` of what is actually a 0-dimensional scalar.

This bug affects any graph where the masked value doesn't appear in the data, which is a common scenario. For example:
- Masking zero values in a graph with all positive weights
- Masking negative values in a graph with only non-negative weights
- Masking infinite values in a finite graph

The error message "axis 1 is out of bounds for array of dimension 0" is cryptic and doesn't help users understand that the issue is with the mask format, not their data.

## Relevant Context

The crash occurs at line 78 in `scipy/sparse/csgraph/_tools.pyx` when the function tries to perform operations on the mask attribute assuming it's always an array. The NumPy documentation for MaskedArray explicitly mentions that masks can be scalar when optimized:

> When no values are masked, `ma.nomask` is often used, which is equivalent to the scalar `False`.

The function `np.ma.getmaskarray()` exists specifically to handle this case - it always returns an array form of the mask, converting scalar `False` to an array of `False` values with the same shape as the data.

From the traceback, the error originates from trying to sum along axis 1 of the mask, which fails when the mask is a scalar.

## Proposed Fix

The fix should handle scalar masks before attempting array operations. The simplest approach is to use NumPy's built-in `getmaskarray` function which handles this conversion:

```diff
--- a/scipy/sparse/csgraph/_tools.pyx
+++ b/scipy/sparse/csgraph/_tools.pyx
@@ -71,7 +71,7 @@ def csgraph_from_masked(graph):

     # ... validation code ...

-    mask = graph.mask
+    mask = np.ma.getmaskarray(graph)

     # Now mask is guaranteed to be an array with the same shape as graph.data
     # ... rest of the function continues normally ...
```

Alternatively, manually handle the scalar case:

```diff
--- a/scipy/sparse/csgraph/_tools.pyx
+++ b/scipy/sparse/csgraph/_tools.pyx
@@ -71,7 +71,15 @@ def csgraph_from_masked(graph):

     # ... validation code ...

-    mask = graph.mask
+    # Handle scalar mask (when no values are masked)
+    if np.ndim(graph.mask) == 0:
+        if graph.mask:
+            # All values are masked
+            mask = np.ones_like(graph.data, dtype=bool)
+        else:
+            # No values are masked
+            mask = np.zeros_like(graph.data, dtype=bool)
+    else:
+        mask = graph.mask

     # ... rest of the function continues normally ...
```