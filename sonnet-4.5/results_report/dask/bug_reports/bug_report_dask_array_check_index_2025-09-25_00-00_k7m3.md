# Bug Report: dask.array.slicing.check_index Misleading Error Message

**Target**: `dask.array.slicing.check_index`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `check_index` function in `dask.array.slicing` raises an error when a boolean index array size doesn't match the expected dimension size. However, the error message incorrectly says the array is "not long enough" even when the array is actually too long (larger than the dimension size).

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
from dask.array.slicing import check_index

@given(st.integers(min_value=1, max_value=100))
def test_check_index_error_message_accuracy(dim_size):
    too_long_array = np.array([True] * (dim_size + 1))

    try:
        check_index(0, too_long_array, dim_size)
        assert False, "Should have raised IndexError"
    except IndexError as e:
        error_msg = str(e)

        if "not long enough" in error_msg and too_long_array.size > dim_size:
            raise AssertionError(
                f"Error message says 'not long enough' but array size "
                f"{too_long_array.size} is greater than dimension {dim_size}"
            )
```

**Failing input**: `dim_size = 1` (or any value), `bool_array` with size > `dim_size`

## Reproducing the Bug

```python
from dask.array.slicing import check_index
import numpy as np

bool_array = np.array([True, True, True])
dimension = 1

try:
    check_index(0, bool_array, dimension)
except IndexError as e:
    print(e)
```

**Output:**
```
IndexError: Boolean array with size 3 is not long enough for axis 0 with size 1
```

The array has size 3 and the dimension has size 1, so the array is **too long**, not "not long enough".

## Why This Is A Bug

This violates the API contract by providing misleading error messages. The error message claims the boolean array is "not long enough" when it may actually be too long. This confuses users trying to debug their code, as they might try to make the array longer when they actually need to make it shorter (or vice versa).

## Fix

```diff
--- a/dask/array/slicing.py
+++ b/dask/array/slicing.py
@@ -917,10 +917,15 @@ def check_index(axis, ind, dimension):
     elif is_arraylike(ind):
         if ind.dtype == bool:
             if ind.size != dimension:
+                if ind.size < dimension:
+                    msg = (
+                        f"Boolean array with size {ind.size} is too short "
+                        f"for axis {axis} with size {dimension}"
+                    )
+                else:
+                    msg = (
+                        f"Boolean array with size {ind.size} is too long "
+                        f"for axis {axis} with size {dimension}"
+                    )
-                raise IndexError(
-                    f"Boolean array with size {ind.size} is not long enough "
-                    f"for axis {axis} with size {dimension}"
-                )
+                raise IndexError(msg)
         elif (ind >= dimension).any() or (ind < -dimension).any():
             raise IndexError(
```