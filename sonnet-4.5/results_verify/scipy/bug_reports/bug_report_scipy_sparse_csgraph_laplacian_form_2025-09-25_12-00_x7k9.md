# Bug Report: scipy.sparse.csgraph.laplacian form='array' Returns Sparse Matrix

**Target**: `scipy.sparse.csgraph.laplacian`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `laplacian` function with `form='array'` returns a sparse matrix instead of a numpy array, violating its documented API contract.

## Property-Based Test

```python
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
import scipy.sparse as sp
from hypothesis import given, strategies as st


@st.composite
def simple_graphs(draw):
    n = draw(st.integers(min_value=2, max_value=5))
    graph = csr_matrix((n, n), dtype=float)

    for i in range(n-1):
        graph[i, i+1] = 1.0
        graph[i+1, i] = 1.0

    return graph


@given(simple_graphs())
def test_laplacian_form_array_returns_numpy_array(graph):
    lap = laplacian(graph, normed=False, form='array')

    assert not sp.issparse(lap), \
        f"laplacian(form='array') should return numpy array, got {type(lap).__name__}"
```

**Failing input**: A 2x2 undirected graph with one edge

## Reproducing the Bug

```python
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
import scipy.sparse as sp

graph = csr_matrix([[0, 1], [1, 0]], dtype=float)

lap = laplacian(graph, normed=False, form='array')

print(f"Type: {type(lap)}")
print(f"Is sparse: {sp.issparse(lap)}")

assert not sp.issparse(lap), f"Expected numpy array, got {type(lap).__name__}"
```

## Why This Is A Bug

The docstring for `scipy.sparse.csgraph.laplacian` explicitly states:

```
form: 'array', or 'function', or 'lo'
    Determines the format of the output Laplacian:

    * 'array' is a numpy array;
    * 'function' is a pointer to evaluating the Laplacian-vector
      or Laplacian-matrix product;
    * 'lo' results in the format of the `LinearOperator`.
```

The documentation clearly states that `form='array'` should return "a numpy array", but the function is returning a `scipy.sparse.coo_matrix` instead. This breaks the API contract and can cause downstream code to fail when it expects a dense numpy array but receives a sparse matrix.

## Fix

The laplacian function should convert the sparse matrix to a dense numpy array when `form='array'` is specified. The fix should be in the `_laplacian.py` file, ensuring that the return value is converted to a dense array using `.toarray()` when the form parameter is 'array'.

A potential fix location would be in the function's return statement when `form == 'array'`:

```python
if form == 'array':
    return lap.toarray()
```