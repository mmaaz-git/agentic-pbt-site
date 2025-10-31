# Bug Report: scipy.io.hb_write Negative Value Formatting

**Target**: `scipy.io.hb_write` and `scipy.io.hb_read`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`scipy.io.hb_write` incorrectly formats negative floating-point values in Harwell-Boeing files, causing adjacent negative values to be written without separators. This makes the files unreadable by `scipy.io.hb_read`, violating the round-trip property.

## Property-Based Test

```python
import numpy as np
import scipy.sparse
import tempfile
import os
from hypothesis import given, strategies as st, settings
from hypothesis.strategies import composite
import scipy.io

@composite
def sparse_matrices_for_hb(draw):
    rows = draw(st.integers(min_value=1, max_value=20))
    cols = draw(st.integers(min_value=1, max_value=20))
    density = draw(st.floats(min_value=0.1, max_value=0.5))
    nnz = max(1, min(int(rows * cols * density), rows * cols))
    row_indices = [draw(st.integers(min_value=0, max_value=rows-1)) for _ in range(nnz)]
    col_indices = [draw(st.integers(min_value=0, max_value=cols-1)) for _ in range(nnz)]
    data = [draw(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False))
            for _ in range(nnz)]
    matrix = scipy.sparse.coo_matrix((data, (row_indices, col_indices)), shape=(rows, cols))
    return matrix.tocsc()

@given(matrix=sparse_matrices_for_hb())
@settings(max_examples=100)
def test_hb_write_read_round_trip(matrix):
    with tempfile.NamedTemporaryFile(suffix='.hb', delete=False, mode='w') as f:
        temp_path = f.name
    try:
        scipy.io.hb_write(temp_path, matrix)
        read_matrix = scipy.io.hb_read(temp_path, spmatrix=True)
        assert read_matrix.shape == matrix.shape
        assert np.allclose(read_matrix.toarray(), matrix.toarray(), rtol=1e-6, atol=1e-10)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
```

**Failing input**: Sparse CSC matrix with negative values, e.g., `scipy.sparse.csc_matrix(([-0.0, -0.0], ([0, 1], [0, 1])), shape=(3, 7))`

## Reproducing the Bug

```python
import numpy as np
import scipy.sparse
import scipy.io
import tempfile
import os

matrix = scipy.sparse.csc_matrix(([-0.0, -0.0], ([0, 1], [0, 1])), shape=(3, 7))

with tempfile.NamedTemporaryFile(suffix='.hb', delete=False, mode='w') as f:
    temp_path = f.name

try:
    scipy.io.hb_write(temp_path, matrix)

    with open(temp_path, 'r') as f:
        content = f.read()
        print("File content:")
        print(content)

    read_matrix = scipy.io.hb_read(temp_path, spmatrix=True)
except ValueError as e:
    print(f"Error: {e}")
finally:
    if os.path.exists(temp_path):
        os.unlink(temp_path)
```

Output shows the problematic line:
```
-0.0000000000000000E+00-0.0000000000000000E+00
```

The two values are concatenated without a space, making them unparseable.

## Why This Is A Bug

The Harwell-Boeing format specification requires values to be space-separated. The format string `(3E24.16)` allocates 24 characters per value, but negative values in scientific notation can use all 24 characters (e.g., `-0.0000000000000000E+00` is exactly 24 chars), leaving no room for separators between adjacent values.

When `hb_read` tries to parse the values using `np.fromstring(..., sep=' ')`, it fails because there's no separator between consecutive negative values.

This violates the fundamental round-trip property: `hb_read(hb_write(matrix)) == matrix`.

## Fix

The format string needs to allocate at least 25 characters per value to ensure proper spacing, or add explicit separators between values. The fix should be in the `hb_write` function where it generates the format string for floating-point values.

A potential fix would be to change the format width from 24 to 25 (or higher) in the relevant part of `scipy/io/_harwell_boeing/hb.py` where the format string is constructed, ensuring negative values always have a separator.