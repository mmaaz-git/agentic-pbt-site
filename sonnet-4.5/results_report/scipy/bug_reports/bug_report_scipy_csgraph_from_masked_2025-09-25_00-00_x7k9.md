# Bug Report: scipy.sparse.csgraph.csgraph_from_masked Crashes on Unmasked Arrays

**Target**: `scipy.sparse.csgraph.csgraph_from_masked`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`csgraph_from_masked` crashes with `AxisError` when given a masked array with no masked values, because NumPy optimizes the mask to a scalar `False` instead of an array.

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
```

**Failing input**: `graph=array([[1., 1.], [1., 1.]])`

## Reproducing the Bug

```python
import numpy as np
from scipy.sparse.csgraph import csgraph_from_masked

graph = np.array([[1., 1.], [1., 1.]])
masked_graph = np.ma.masked_equal(graph, 0)

print(f"Mask: {masked_graph.mask}")
print(f"Mask shape: {masked_graph.mask.shape}")
print(f"Mask ndim: {masked_graph.mask.ndim}")

sparse_graph = csgraph_from_masked(masked_graph)
```

Output:
```
Mask: False
Mask shape: ()
Mask ndim: 0

numpy.exceptions.AxisError: axis 1 is out of bounds for array of dimension 0
```

## Why This Is A Bug

When a masked array has no masked elements, NumPy optimizes the mask attribute to a scalar `False` instead of an array of `False` values. The `csgraph_from_masked` function attempts to operate on this mask as if it were always an array, leading to an `AxisError` when it tries to access `axis=1` on a 0-dimensional scalar.

This is valid input according to the documented interface - any masked array should be accepted. Users should be able to convert any masked graph representation to a sparse graph without crashes.

## Fix

The fix should normalize the mask to always be an array before processing. In the Cython source file `scipy/sparse/csgraph/_tools.pyx`, the function should handle scalar masks:

```diff
def csgraph_from_masked(graph):
    # ... existing validation code ...
+
+   # Handle scalar mask (when no values are masked)
+   if np.ndim(graph.mask) == 0:
+       if graph.mask:
+           # All values are masked - create empty sparse array
+           graph.mask = np.ones_like(graph.data, dtype=bool)
+       else:
+           # No values are masked - treat as fully unmasked
+           graph.mask = np.zeros_like(graph.data, dtype=bool)
+
    # ... rest of the existing code ...
```

Alternatively, use `np.ma.getmaskarray(graph)` which always returns an array-form mask, even when the mask is a scalar `False`.