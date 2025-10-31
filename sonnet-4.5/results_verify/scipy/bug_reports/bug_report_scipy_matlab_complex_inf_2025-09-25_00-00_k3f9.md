# Bug Report: scipy.io.matlab Complex Infinity Corruption

**Target**: `scipy.io.matlab.savemat` / `scipy.io.matlab.loadmat`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Complex numbers with zero real part and infinite imaginary part are corrupted during MATLAB format 4 save/load round-trip. The real part changes from 0 to NaN.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from hypothesis.extra.numpy import arrays, array_shapes
import scipy.io.matlab as sio_matlab
import numpy as np
import tempfile
import os

@given(
    data=st.dictionaries(
        keys=st.from_regex(r'[a-zA-Z][a-zA-Z0-9_]{0,30}', fullmatch=True),
        values=arrays(
            dtype=st.sampled_from([np.float64, np.int32, np.uint8, np.complex128]),
            shape=array_shapes(max_dims=2, max_side=20),
        ),
        min_size=1,
        max_size=5
    )
)
@settings(max_examples=100)
def test_savemat_loadmat_roundtrip_format4(data):
    with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
        fname = f.name

    try:
        sio_matlab.savemat(fname, data, format='4')
        loaded = sio_matlab.loadmat(fname)

        for key in data.keys():
            assert key in loaded
            original = data[key]
            result = loaded[key]

            if original.ndim == 0:
                expected_shape = (1, 1)
            elif original.ndim == 1:
                expected_shape = (1, original.shape[0])
            else:
                expected_shape = original.shape

            assert result.shape == expected_shape

            original_reshaped = original.reshape(expected_shape)
            np.testing.assert_array_equal(result, original_reshaped)
    finally:
        if os.path.exists(fname):
            os.remove(fname)
```

**Failing input**: `data={'A': array([0.+infj])}`

## Reproducing the Bug

```python
import numpy as np
import scipy.io.matlab as sio
from io import BytesIO

original = np.array([0.0 + 1j * np.inf])
print(f"Original: {original}")

f = BytesIO()
sio.savemat(f, {'x': original}, format='4')
loaded = sio.loadmat(f)
result = loaded['x']

print(f"Loaded: {result}")
print(f"Bug: real part is {result[0,0].real} (expected 0.0)")
```

Output:
```
Original: [0.+infj]
Loaded: [[nan+infj]]
Bug: real part is nan (expected 0.0)
```

## Why This Is A Bug

The documentation for `savemat` and `loadmat` states that `struct_as_record=True` (the default) "allows easier round-trip load and save of MATLAB files." This implies that data should be preserved during a save/load cycle. However, complex numbers with zero real part and infinite imaginary part are corrupted - the real part changes from 0 to NaN.

This violates the fundamental expectation that `loadmat(savemat(data))` should preserve numeric values, and constitutes silent data corruption. Users expect IEEE 754 special values (infinity) to be handled correctly.

## Fix

The bug is in `/scipy/io/matlab/_mio4.py` at line 218. The code reads complex arrays by:

```python
res = self.read_sub_array(hdr, copy=False)
res_j = self.read_sub_array(hdr, copy=False)
return res + (res_j * 1j)
```

The issue occurs because of the `copy=False` parameter. When both the real and imaginary parts use `copy=False`, and special float values like infinity are involved, NumPy array operations can produce unexpected results due to memory aliasing or undefined behavior in the underlying operation.

The fix is to ensure at least one of the arrays is copied:

```diff
--- a/scipy/io/matlab/_mio4.py
+++ b/scipy/io/matlab/_mio4.py
@@ -213,8 +213,8 @@ class MatFile4Reader(MatFileReader):
             with dtype ``float`` and shape given by `hdr` ``dims``
         '''
         if hdr.is_complex:
-            # avoid array copy to save memory
-            res = self.read_sub_array(hdr, copy=False)
+            # need to copy to avoid corruption with special float values
+            res = self.read_sub_array(hdr, copy=True)
             res_j = self.read_sub_array(hdr, copy=False)
             return res + (res_j * 1j)
         return self.read_sub_array(hdr)
```

Alternatively, the operation could be changed to avoid in-place operations:

```diff
--- a/scipy/io/matlab/_mio4.py
+++ b/scipy/io/matlab/_mio4.py
@@ -215,7 +215,7 @@ class MatFile4Reader(MatFileReader):
         if hdr.is_complex:
             res = self.read_sub_array(hdr, copy=False)
             res_j = self.read_sub_array(hdr, copy=False)
-            return res + (res_j * 1j)
+            return np.array(res, dtype=complex) + (res_j * 1j)
         return self.read_sub_array(hdr)
```