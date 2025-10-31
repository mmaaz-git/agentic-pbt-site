# Bug Report: numpy.matrixlib.matrix Reduction Methods Ignore out Parameter Return Value

**Target**: `numpy.matrixlib.defmatrix.matrix` (methods: `sum`, `mean`, `std`, `var`, `prod`, `max`, `min`)
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

When `axis=None`, matrix reduction methods (`sum`, `mean`, `std`, `var`, `prod`, `max`, `min`) populate the `out` parameter but return a scalar instead of returning the `out` parameter, violating NumPy's API contract.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from hypothesis.extra.numpy import arrays
import numpy as np
from numpy import matrix


@given(arrays(np.float64, shape=st.tuples(st.integers(3, 20), st.integers(3, 20)),
              elements=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)))
@settings(max_examples=500)
def test_sum_returns_out_parameter(arr):
    m = matrix(arr)
    out = matrix([[0.0]])
    result = m.sum(axis=None, out=out)
    assert result is out, "sum should return out parameter"
```

**Failing input**: Any matrix

## Reproducing the Bug

```python
import numpy as np
from numpy import matrix

m = matrix([[1, 2], [3, 4]])
out = matrix([[0.0]])
result = m.sum(axis=None, out=out)

print(f"Result: {result}, type: {type(result)}")
print(f"Out: {out}, type: {type(out)}")
print(f"Result is out: {result is out}")
```

**Output**:
```
Result: 10.0, type: <class 'numpy.float64'>
Out: [[10.]], type: <class 'numpy.matrix'>
Result is out: False
```

The `out` parameter is populated with the correct value (10.0), but the function returns a scalar instead of the `out` parameter.

## Why This Is A Bug

NumPy's API contract states that when a function accepts an `out` parameter, it should return that parameter. This allows for method chaining and is the expected behavior across NumPy's API. The current implementation violates this contract when `axis=None`.

The issue is in the `_collapse` method (line 261-268):

```python
def _collapse(self, axis):
    if axis is None:
        return self[0, 0]  # Returns scalar, losing the reference to out
    else:
        return self
```

When `axis=None`, `_collapse` extracts the scalar value, discarding the matrix object that contains the `out` parameter.

## Fix

The fix requires modifying the `_collapse` method to preserve the `out` parameter when present. However, this is non-trivial because `_collapse` doesn't have access to the original `out` parameter. A simpler fix would be to document this behavior, or to check if the result came from an `out` parameter and return accordingly.

Alternatively, each affected method could be modified to handle the `axis=None` case specially:

```diff
--- a/numpy/matrixlib/defmatrix.py
+++ b/numpy/matrixlib/defmatrix.py
@@ -324,7 +324,11 @@ class matrix(N.ndarray):
                 [7.]])

         """
-        return N.ndarray.sum(self, axis, dtype, out, keepdims=True)._collapse(axis)
+        result = N.ndarray.sum(self, axis, dtype, out, keepdims=True)
+        if axis is None and out is not None:
+            out[()] = result[0, 0]
+            return out
+        return result._collapse(axis)
```

This same pattern would need to be applied to `mean`, `std`, `var`, `prod`, `max`, and `min`.