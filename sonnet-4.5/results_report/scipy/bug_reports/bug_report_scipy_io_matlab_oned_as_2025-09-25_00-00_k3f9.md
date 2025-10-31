# Bug Report: scipy.io.matlab savemat oned_as Parameter Ignored for Empty Arrays

**Target**: `scipy.io.matlab.savemat`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `oned_as` parameter in `scipy.io.matlab.savemat` is ignored when saving empty 1D arrays, resulting in inconsistent behavior compared to non-empty 1D arrays. Both `oned_as='row'` and `oned_as='column'` produce a `(0, 0)` shaped array when the input is an empty 1D array `(0,)`, instead of the expected `(1, 0)` and `(0, 1)` shapes respectively.

## Property-Based Test

```python
import numpy as np
from io import BytesIO
from hypothesis import given, strategies as st
import scipy.io.matlab as matlab


@given(oned_as=st.sampled_from(['row', 'column']))
def test_oned_as_consistency_empty_arrays(oned_as):
    arr = np.array([])

    f = BytesIO()
    matlab.savemat(f, {'arr': arr}, oned_as=oned_as)
    f.seek(0)
    loaded = matlab.loadmat(f)
    result = loaded['arr']

    if oned_as == 'row':
        expected_shape = (1, 0)
    else:
        expected_shape = (0, 1)

    assert result.shape == expected_shape, (
        f"oned_as='{oned_as}' should produce shape {expected_shape}, "
        f"but got {result.shape}"
    )
```

**Failing input**: `oned_as='row'` (also fails for `oned_as='column'`)

## Reproducing the Bug

```python
import numpy as np
from io import BytesIO
import scipy.io.matlab as matlab

arr_empty = np.array([])
arr_nonempty = np.array([1, 2, 3])

print(f"Original empty array shape: {arr_empty.shape}")
print(f"Original non-empty array shape: {arr_nonempty.shape}")

for oned_as in ['row', 'column']:
    print(f"\nTesting with oned_as='{oned_as}':")

    f_empty = BytesIO()
    matlab.savemat(f_empty, {'arr': arr_empty}, oned_as=oned_as)
    f_empty.seek(0)
    loaded_empty = matlab.loadmat(f_empty)['arr']

    f_nonempty = BytesIO()
    matlab.savemat(f_nonempty, {'arr': arr_nonempty}, oned_as=oned_as)
    f_nonempty.seek(0)
    loaded_nonempty = matlab.loadmat(f_nonempty)['arr']

    print(f"  Non-empty [1,2,3]: (3,) -> {loaded_nonempty.shape}")
    print(f"  Empty []:          (0,) -> {loaded_empty.shape}")

    if oned_as == 'row':
        print(f"  Expected empty: (1, 0), Got: {loaded_empty.shape}")
    else:
        print(f"  Expected empty: (0, 1), Got: {loaded_empty.shape}")
```

Output:
```
Original empty array shape: (0,)
Original non-empty array shape: (3,)

Testing with oned_as='row':
  Non-empty [1,2,3]: (3,) -> (1, 3)
  Empty []:          (0,) -> (0, 0)
  Expected empty: (1, 0), Got: (0, 0)

Testing with oned_as='column':
  Non-empty [1,2,3]: (3,) -> (3, 1)
  Empty []:          (0,) -> (0, 0)
  Expected empty: (0, 1), Got: (0, 0)
```

## Why This Is A Bug

The `oned_as` parameter is documented to control how 1D NumPy arrays are written:
- `oned_as='row'`: Write 1-D arrays as row vectors (shape `(1, N)`)
- `oned_as='column'`: Write 1-D arrays as column vectors (shape `(N, 1)`)

For non-empty 1D arrays, this works correctly. However, for empty 1D arrays (shape `(0,)`), the parameter is completely ignored and both settings produce `(0, 0)` instead of the expected `(1, 0)` or `(0, 1)`.

This inconsistency:
1. Violates the documented behavior of the `oned_as` parameter
2. Makes it impossible to distinguish between row and column vectors when they are empty
3. Breaks the expected symmetry between empty and non-empty arrays
4. Can cause issues in code that depends on consistent dimensionality

## Fix

The issue likely occurs in the array writing logic where empty arrays are handled as a special case without considering the `oned_as` parameter. The fix should ensure that empty 1D arrays are reshaped according to `oned_as` before writing, consistent with non-empty arrays.

A potential fix would be to explicitly reshape empty 1D arrays in the same way as non-empty arrays:
- For `oned_as='row'`: `arr.reshape(1, -1)` produces `(1, 0)`
- For `oned_as='column'`: `arr.reshape(-1, 1)` produces `(0, 1)`

The exact location of the fix would be in the `scipy/io/matlab/_mio.py` or `scipy/io/matlab/_mio5.py` modules, where 1D array handling logic resides.