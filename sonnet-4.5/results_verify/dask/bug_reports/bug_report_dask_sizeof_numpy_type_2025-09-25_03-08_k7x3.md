# Bug Report: dask.sizeof Return Type Inconsistency for Broadcasted NumPy Arrays

**Target**: `dask.sizeof.sizeof_numpy_ndarray`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `sizeof` function for NumPy arrays returns inconsistent types: Python `int` for regular arrays but `numpy.intp` for broadcasted arrays (arrays with 0 in strides). This violates the implicit contract that `sizeof` returns Python `int` and causes `isinstance(result, int)` checks to fail for broadcasted arrays.

## Property-Based Test

```python
from hypothesis import given
import hypothesis.strategies as st
import hypothesis.extra.numpy as npst
from dask.sizeof import sizeof
import numpy as np

@given(npst.arrays(dtype=np.float64, shape=st.tuples(st.integers(1, 100))))
def test_sizeof_returns_python_int(arr):
    """sizeof should always return Python int, not numpy integer types"""
    result = sizeof(arr)
    assert isinstance(result, int), f"Expected int, got {type(result)}"
    assert type(result) is int, f"Expected Python int, got {type(result).__name__}"
```

**Failing input**: `np.broadcast_to(1, (100, 100))` - any broadcasted NumPy array

## Reproducing the Bug

```python
import numpy as np
from dask.sizeof import sizeof

arr_regular = np.ones(100, dtype='f8')
result_regular = sizeof(arr_regular)

arr_broadcast = np.broadcast_to(1, (100, 100))
result_broadcast = sizeof(arr_broadcast)

print(f"Regular array: type={type(result_regular)} is int? {type(result_regular) is int}")
print(f"Broadcast array: type={type(result_broadcast)} is int? {type(result_broadcast) is int}")

assert type(result_regular) is int
assert type(result_broadcast) is int
```

**Output**:
```
Regular array: type=<class 'int'> is int? True
Broadcast array: type=<class 'numpy.intp'> is int? False
AssertionError
```

## Why This Is A Bug

1. **Contract violation**: The existing test suite (test_sizeof.py lines 65-67) explicitly checks `isinstance(sizeof(...), int)`, establishing that sizeof should return Python int.

2. **Type inconsistency**: Within the same function, the regular path (line 140) wraps the result with `int()`, but the broadcasted array path (line 139) does not.

3. **API expectations**: Users expect `sizeof()` to return Python int for all inputs, not numpy integer types.

4. **Downstream issues**: Code using `type(result) is int` checks will fail for broadcasted arrays.

## Fix

```diff
--- a/dask/sizeof.py
+++ b/dask/sizeof.py
@@ -136,7 +136,7 @@ def register_numpy():
     def sizeof_numpy_ndarray(x):
         if 0 in x.strides:
             xs = x[tuple(slice(None) if s != 0 else slice(1) for s in x.strides)]
-            return xs.nbytes
+            return int(xs.nbytes)
         return int(x.nbytes)
```

The fix adds `int()` wrapper to line 139 to ensure consistent return type with line 140.