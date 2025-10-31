# Bug Report: scipy.io.matlab Empty 1D Array Round-Trip Shape Change

**Target**: `scipy.io.matlab.savemat` / `scipy.io.matlab.loadmat`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Empty 1D arrays `(0,)` are inconsistently converted to `(0, 0)` during savemat/loadmat round-trip, while non-empty 1D arrays respect the `oned_as` parameter. This violates the documented round-trip property and creates inconsistent behavior.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
import tempfile
from scipy.io.matlab import loadmat, savemat

def valid_matlab_varname():
    first = st.sampled_from('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
    rest = st.text(alphabet='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_', min_size=0, max_size=30)
    return st.tuples(first, rest).map(lambda x: x[0] + x[1])

def matlab_compatible_arrays():
    return st.one_of(
        st.integers(min_value=-1000, max_value=1000).map(lambda x: np.array(x)),
        st.lists(st.integers(min_value=-1000, max_value=1000), min_size=0, max_size=20).map(lambda x: np.array(x)),
    )

@given(st.dictionaries(valid_matlab_varname(), matlab_compatible_arrays(), min_size=1, max_size=5))
@settings(max_examples=200)
def test_roundtrip_savemat_loadmat(data_dict):
    with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
        fname = f.name

    try:
        savemat(fname, data_dict)
        loaded = loadmat(fname)

        for key in data_dict:
            assert key in loaded
            np.testing.assert_array_equal(loaded[key], data_dict[key])
    finally:
        import os
        if os.path.exists(fname):
            os.unlink(fname)
```

**Failing input**: `data_dict={'a': np.array([])}`

## Reproducing the Bug

```python
import numpy as np
import tempfile
from scipy.io.matlab import loadmat, savemat

empty_1d = np.array([])
print(f"Original shape: {empty_1d.shape}")

with tempfile.NamedTemporaryFile(suffix='.mat') as f:
    savemat(f.name, {'arr': empty_1d})
    loaded = loadmat(f.name)['arr']
    print(f"Loaded shape: {loaded.shape}")
    print(f"Match: {empty_1d.shape == loaded.shape}")
```

Output:
```
Original shape: (0,)
Loaded shape: (0, 0)
Match: False
```

## Why This Is A Bug

1. **Violates documented round-trip property**: The `loadmat` docstring (line 124 in `_mio.py`) states: "The default setting is True, because it allows easier round-trip load and save of MATLAB files." This bug breaks that guarantee.

2. **Inconsistent behavior**: Non-empty 1D arrays are handled differently than empty ones:
   - Non-empty with `oned_as='column'`: `(n,)` → `(n, 1)`
   - Non-empty with `oned_as='row'`: `(n,)` → `(1, n)`
   - **Empty**: `(0,)` → `(0, 0)` regardless of `oned_as`

3. **Expected behavior**: Empty 1D arrays should follow the same pattern as non-empty ones:
   - With `oned_as='column'`: `(0,)` → `(0, 1)`
   - With `oned_as='row'`: `(0,)` → `(1, 0)`

## Fix

The bug is in `/scipy/io/matlab/_miobase.py` lines 326-327:

```diff
--- a/scipy/io/matlab/_miobase.py
+++ b/scipy/io/matlab/_miobase.py
@@ -323,8 +323,6 @@ def matdims(arr, oned_as='column'):
     if shape == ():  # scalar
         return (1, 1)
     if len(shape) == 1:  # 1D
-        if shape[0] == 0:
-            return (0, 0)
         if oned_as == 'column':
             return shape + (1,)
         elif oned_as == 'row':
```

This change makes empty 1D arrays behave consistently with non-empty ones:
- `(0,)` with `oned_as='column'` becomes `(0, 1)`
- `(0,)` with `oned_as='row'` becomes `(1, 0)`

The corresponding docstring example at line 302-303 should also be updated to reflect the corrected behavior.