# Bug Report: numpy.matrixlib.matrix - Inconsistent 3D Input Handling

**Target**: `numpy.matrixlib.matrix`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `matrix` constructor handles 3D input inconsistently: 3D numpy arrays may succeed (after dimension squeezing) while equivalent 3D lists always fail, violating the principle that behavior should be input-type independent.

## Property-Based Test

```python
import numpy as np
from numpy.matrixlib import matrix
from hypothesis import given, strategies as st
import pytest


@given(st.integers(1, 5), st.integers(2, 5))
def test_3d_array_vs_list_consistency(n, m):
    shape_3d = (1, n, m)

    arr_3d = np.zeros(shape_3d)
    list_3d = [[[0.0] * m for _ in range(n)]]

    arr_result = None
    arr_error = None
    try:
        arr_result = matrix(arr_3d)
    except ValueError as e:
        arr_error = str(e)

    list_result = None
    list_error = None
    try:
        list_result = matrix(list_3d)
    except ValueError as e:
        list_error = str(e)

    if arr_result is None and list_result is None:
        assert arr_error == list_error, f"Both failed but with different errors: '{arr_error}' vs '{list_error}'"
    elif arr_result is not None and list_result is not None:
        assert arr_result.shape == list_result.shape, "Both succeeded but with different shapes"
    else:
        arr_status = 'succeeded' if arr_result is not None else 'failed'
        list_status = 'succeeded' if list_result is not None else 'failed'
        pytest.fail(f"Inconsistent behavior: array {arr_status}, list {list_status}")
```

**Failing input**: `n=1, m=2` (shape `(1,1,2)`)

## Reproducing the Bug

```python
import numpy as np
from numpy.matrixlib import matrix

arr_3d = np.zeros((1, 1, 2))
list_3d = [[[0.0, 0.0]]]

m1 = matrix(arr_3d)
print(f"3D array -> matrix: SUCCESS, shape = {m1.shape}")

m2 = matrix(list_3d)
print(f"3D list -> matrix: FAILED")
```

**Output:**
```
3D array -> matrix: SUCCESS, shape = (1, 2)
3D list -> matrix: ValueError: matrix must be 2-dimensional
```

## Why This Is A Bug

The matrix constructor has two code paths with different dimension-handling logic:

1. **ndarray input** (lines 134-145): Uses `.view(subtype)` → triggers `__array_finalize__` → squeezes size-1 dimensions → may succeed for 3D arrays like `(1,1,2)` → shape `(1,2)`

2. **list input** (lines 147-172): Converts to ndarray → checks `ndim > 2` → always rejects 3D input → error: "matrix must be 2-dimensional"

This violates the API contract that logically equivalent inputs (3D array vs 3D list with same structure) should behave identically. Users expect `matrix(arr.tolist())` to behave like `matrix(arr)`.

## Fix

Standardize dimension validation by checking dimensions before any view operations:

```diff
--- a/numpy/matrixlib/defmatrix.py
+++ b/numpy/matrixlib/defmatrix.py
@@ -133,6 +133,10 @@ class matrix(N.ndarray):

         if isinstance(data, N.ndarray):
+            if data.ndim > 2:
+                raise ValueError("matrix must be 2-dimensional")
+
             if dtype is None:
                 intype = data.dtype
             else:
```