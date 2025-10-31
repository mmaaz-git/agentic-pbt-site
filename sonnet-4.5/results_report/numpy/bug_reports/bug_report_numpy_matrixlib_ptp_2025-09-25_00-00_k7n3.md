# Bug Report: numpy.matrixlib.matrix.ptp Crashes with out Parameter

**Target**: `numpy.matrixlib.defmatrix.matrix.ptp`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `matrix.ptp()` method crashes with a ValueError when called with an `out` parameter, regardless of the axis value.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from hypothesis.extra.numpy import arrays
import numpy as np
from numpy import matrix


@given(arrays(np.float64, shape=st.tuples(st.integers(3, 20), st.integers(3, 20)),
              elements=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)))
@settings(max_examples=300)
def test_ptp_with_out_parameter(arr):
    m = matrix(arr)
    out = matrix(np.zeros((1, 1)))
    result = m.ptp(axis=None, out=out)
    assert result is out
```

**Failing input**: Any matrix with `out` parameter

## Reproducing the Bug

```python
import numpy as np
from numpy import matrix

m = matrix([[1, 2], [3, 4]])
out = matrix([[0.0]])
result = m.ptp(axis=None, out=out)
```

**Output**:
```
ValueError: output parameter for reduction operation maximum has the wrong number of dimensions: Found 2 but expected 0
```

## Why This Is A Bug

The `ptp` method should accept an `out` parameter consistent with NumPy's API contract and other matrix methods. The crash occurs because `ptp` calls `N.ptp(self, axis, out)` without `keepdims=True`, causing the underlying NumPy function to expect a 0-dimensional output when `axis=None`, but receives a 2-dimensional matrix.

All other reduction methods (`sum`, `mean`, `std`, `var`, `prod`, `max`, `min`) call the underlying NumPy function with `keepdims=True` to maintain matrix dimensionality, but `ptp` does not.

## Fix

```diff
--- a/numpy/matrixlib/defmatrix.py
+++ b/numpy/matrixlib/defmatrix.py
@@ -795,7 +795,7 @@ class matrix(N.ndarray):
                 [3]])

         """
-        return N.ptp(self, axis, out)._align(axis)
+        return N.ptp(self, axis, out, keepdims=True)._align(axis)
```