# Bug Report: numpy.lib.format.dtype_to_descr Loses Subdtype Information

**Target**: `numpy.lib.format.dtype_to_descr`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`numpy.lib.format.dtype_to_descr` fails to preserve shape information for subdtypes (dtypes with shapes), violating its documented contract that the result can be passed to `numpy.dtype()` to replicate the input dtype.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
import numpy.lib.format

@given(
    shape=st.lists(st.integers(min_value=1, max_value=5), min_size=1, max_size=3)
)
def test_dtype_round_trip_with_shape(shape):
    dtype = np.dtype((np.float32, tuple(shape)))
    descr = np.lib.format.dtype_to_descr(dtype)
    reconstructed = np.lib.format.descr_to_dtype(descr)

    assert reconstructed == dtype
```

**Failing input**: `shape=[1]` (or any shape)

## Reproducing the Bug

```python
import numpy as np
import numpy.lib.format

dtype = np.dtype((np.float32, (2, 3)))
print(f"Original dtype: {dtype}")
print(f"  subdtype: {dtype.subdtype}")

descr = np.lib.format.dtype_to_descr(dtype)
print(f"dtype_to_descr result: {descr}")

reconstructed = np.lib.format.descr_to_dtype(descr)
print(f"descr_to_dtype result: {reconstructed}")

print(f"Original == Reconstructed? {dtype == reconstructed}")
```

Output:
```
Original dtype: ('<f4', (2, 3))
  subdtype: (dtype('float32'), (2, 3))
dtype_to_descr result: |V24
descr_to_dtype result: |V24
Original == Reconstructed? False
```

## Why This Is A Bug

The function's docstring explicitly states:

> Returns
> -------
> descr : object
>     An object that can be passed to `numpy.dtype()` in order to
>     replicate the input dtype.

For subdtypes (dtypes with shapes like `dtype((np.float32, (2, 3)))`), the function returns `dtype.str` (e.g., `|V24`), which creates a void dtype that cannot replicate the original subdtype structure. The subdtype information (base dtype + shape) is completely lost.

## Fix

```diff
--- a/numpy/lib/_format_impl.py
+++ b/numpy/lib/_format_impl.py
@@ -120,6 +120,10 @@ def dtype_to_descr(dtype):
                       UserWarning, stacklevel=2)
     dtype = new_dtype

+    if dtype.subdtype is not None:
+        # Handle subdtypes (dtypes with shapes)
+        base_dtype, shape = dtype.subdtype
+        return (dtype_to_descr(base_dtype), shape)
     if dtype.names is not None:
         # This is a record array. The .descr is fine.  XXX: parts of the
         # record array with an empty name, like padding bytes, still get
```
