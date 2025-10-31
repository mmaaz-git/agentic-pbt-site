# Bug Report: numpy.lib.format.dtype_to_descr Sub-Array Dtype Round-Trip Failure

**Target**: `numpy.lib.format.dtype_to_descr` and `numpy.lib.format.descr_to_dtype`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `dtype_to_descr` function loses shape information for sub-array dtypes, violating its documented contract that the result "can be passed to `numpy.dtype()` in order to replicate the input dtype". The round-trip `descr_to_dtype(dtype_to_descr(dtype))` fails for dtypes with shapes.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st

@given(
    base_type=st.sampled_from(['i4', 'f8', 'c16']),
    shape_size=st.integers(min_value=1, max_value=5)
)
def test_dtype_descr_round_trip_with_shapes(base_type, shape_size):
    shape = tuple(range(1, shape_size + 1))
    dtype = np.dtype((base_type, shape))

    descr = np.lib.format.dtype_to_descr(dtype)
    restored = np.lib.format.descr_to_dtype(descr)

    assert restored == dtype, \
        f"Round-trip failed: {dtype} -> {descr} -> {restored}"
```

**Failing input**: `dtype=np.dtype(('i4', (1,)))`

## Reproducing the Bug

```python
import numpy as np
import numpy.lib.format as fmt

dtype_with_shape = np.dtype(('i4', (1,)))
print(f"Original dtype: {dtype_with_shape}")
print(f"Original shape: {dtype_with_shape.shape}")

descr = fmt.dtype_to_descr(dtype_with_shape)
print(f"After dtype_to_descr: {descr}")

restored = fmt.descr_to_dtype(descr)
print(f"Restored dtype: {restored}")
print(f"Restored shape: {restored.shape}")

assert restored == dtype_with_shape
```

**Output:**
```
Original dtype: ('<i4', (1,))
Original shape: (1,)
After dtype_to_descr: |V4
Restored dtype: |V4
Restored shape: ()
AssertionError
```

## Why This Is A Bug

The docstring for `dtype_to_descr` explicitly states:

> Returns an object that can be passed to `numpy.dtype()` in order to replicate the input dtype.

The docstring for `descr_to_dtype` states:

> This is essentially the reverse of `~lib.format.dtype_to_descr`.

However, for sub-array dtypes (dtypes with a shape attribute), the function returns `dtype.str`, which is a string like `"|V4"` that only contains the total byte size, completely losing the shape information. When `descr_to_dtype` processes this string, it cannot recover the original dtype structure.

This violates the documented API contract that these functions should be inverses of each other.

## Fix

The issue is in `dtype_to_descr` at line 565-567 in `/home/npc/miniconda/lib/python3.13/site-packages/numpy/lib/format.py`:

```python
else:
    return dtype.str
```

For sub-array dtypes, we need to preserve the shape information. The fix should return a tuple containing both the base dtype and shape:

```diff
--- a/numpy/lib/format.py
+++ b/numpy/lib/format.py
@@ -564,7 +564,10 @@ def dtype_to_descr(dtype):
                       "allow_pickle=True to be set.",
                       UserWarning, stacklevel=2)
         return "|O"
-    else:
+    elif dtype.shape == ():
         return dtype.str
+    else:
+        # Sub-array dtype: preserve shape information
+        return (dtype.base.str, dtype.shape)
```

This ensures that sub-array dtypes can be properly round-tripped through the descr format.