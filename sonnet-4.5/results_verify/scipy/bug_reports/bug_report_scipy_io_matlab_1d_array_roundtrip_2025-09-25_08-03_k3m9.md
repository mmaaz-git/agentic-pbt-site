# Bug Report: scipy.io.matlab 1D Array Round-Trip Shape Loss

**Target**: `scipy.io.matlab.savemat` and `scipy.io.matlab.loadmat`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

1D NumPy arrays lose their dimensionality when saved with `savemat` and loaded with `loadmat`, becoming 2D arrays. This breaks the fundamental round-trip property and violates reasonable user expectations, with no complete workaround available.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from hypothesis.extra import numpy as npst
import numpy as np
import scipy.io.matlab as matlab
import tempfile
import os

@given(
    npst.arrays(
        dtype=npst.floating_dtypes(),
        shape=npst.array_shapes(min_dims=1, max_dims=1, min_side=1, max_side=10)
    )
)
@settings(max_examples=100)
def test_roundtrip_1d_arrays(arr):
    assume(not np.any(np.isnan(arr)) and not np.any(np.isinf(arr)))

    with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
        temp_filename = f.name

    try:
        data_dict = {'test_array': arr}
        matlab.savemat(temp_filename, data_dict)
        loaded_dict = matlab.loadmat(temp_filename)

        assert 'test_array' in loaded_dict
        assert loaded_dict['test_array'].shape == arr.shape, \
            f"Shape mismatch: original {arr.shape}, loaded {loaded_dict['test_array'].shape}"
    finally:
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)
```

**Failing input**: `arr=array([0.], dtype=float16)` (any 1D array fails)

## Reproducing the Bug

```python
import numpy as np
import scipy.io.matlab as matlab
import tempfile

arr_1d = np.array([1.0, 2.0, 3.0])
print(f"Original: shape={arr_1d.shape}, ndim={arr_1d.ndim}")

with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
    matlab.savemat(f.name, {'data': arr_1d})
    loaded = matlab.loadmat(f.name)
    loaded_arr = loaded['data']

print(f"Loaded:   shape={loaded_arr.shape}, ndim={loaded_arr.ndim}")
print(f"Shapes match: {arr_1d.shape == loaded_arr.shape}")
```

Output:
```
Original: shape=(3,), ndim=1
Loaded:   shape=(1, 3), ndim=2
Shapes match: False
```

## Why This Is A Bug

1. **Round-trip property violation**: The fundamental expectation that `loadmat(savemat({'x': arr}))['x']` preserves array properties is broken for 1D arrays.

2. **Inadequate documentation**: `loadmat` docstring does not warn users that 1D arrays become 2D, leading to unexpected behavior.

3. **Inconsistent workaround**: The `squeeze_me=True` parameter can partially address this:
   - Works for multi-element 1D arrays: `shape=(5,)` → save/load → `shape=(5,)`
   - Fails for single-element 1D arrays: `shape=(1,)` → save/load with `squeeze_me=True` → becomes scalar `float`, not array

4. **User expectation**: NumPy users regularly work with 1D arrays and reasonably expect shape preservation through serialization.

## Fix

### Option 1: Add unsqueeze parameter to loadmat

```diff
--- a/scipy/io/matlab/_mio.py
+++ b/scipy/io/matlab/_mio.py
@@ -147,7 +147,7 @@ def loadmat(file_name, mdict=None, appendmat=True, *, spmatrix=True, **kwargs):
     file_name : str
        Name of the mat file (do not need .mat extension if
        appendmat==True). Can also pass open file-like object.
     ...
+    restore_1d : bool, optional
+       Whether to restore 1D arrays that were saved as 2D row or column vectors.
+       Default is False for backward compatibility.
     ...
```

And implement logic to detect and restore 1D arrays based on shape (either `(1, N)` or `(N, 1)` with appropriate heuristics).

### Option 2: Document the limitation clearly

Add prominent warning to both `savemat` and `loadmat` docstrings:

```diff
--- a/scipy/io/matlab/_mio.py
+++ b/scipy/io/matlab/_mio.py
@@ -220,6 +220,11 @@ def savemat(file_name, mdict, appendmat=True, format='5',
     ...
+    .. warning::
+       1-D NumPy arrays will be converted to 2-D arrays when saved (as row or
+       column vectors, controlled by `oned_as`). They will remain 2-D when
+       loaded back with `loadmat`. Use `squeeze_me=True` when loading to restore
+       1-D shape for multi-element arrays (single-element arrays become scalars).
     ...
```

### Recommended Approach

Implement Option 1 with `restore_1d=False` by default for backward compatibility, and also add documentation from Option 2.