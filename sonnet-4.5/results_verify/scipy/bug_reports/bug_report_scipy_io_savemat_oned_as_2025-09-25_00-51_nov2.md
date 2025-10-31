# Bug Report: scipy.io.savemat oned_as Parameter Ignored for Empty 1D Arrays

**Target**: `scipy.io.savemat` and `scipy.io.loadmat`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `oned_as` parameter in `scipy.io.savemat` is ignored for empty 1D arrays, causing inconsistent round-trip behavior. Non-empty 1D arrays correctly respect the `oned_as` parameter ('row' → (1, n), 'column' → (n, 1)), but empty 1D arrays always become (0, 0) regardless of the setting, when they should become (1, 0) or (0, 1).

## Property-Based Test

```python
import tempfile
import os
import numpy as np
from scipy import io
from hypothesis import given, strategies as st, settings


def make_matlab_compatible_dict():
    valid_name = st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll'), min_codepoint=65, max_codepoint=122),
        min_size=1,
        max_size=10
    ).filter(lambda x: x[0].isalpha() and x.isidentifier())

    value_strategy = st.one_of(
        st.floats(min_value=-1e10, max_value=1e10, allow_nan=False, allow_infinity=False),
        st.integers(min_value=-1000000, max_value=1000000),
        st.lists(st.floats(min_value=-1e10, max_value=1e10, allow_nan=False, allow_infinity=False),
                 min_size=0, max_size=10),
    )

    return st.dictionaries(valid_name, value_strategy, min_size=1, max_size=5)


@given(make_matlab_compatible_dict())
@settings(max_examples=50)
def test_savemat_loadmat_roundtrip(data_dict):
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.mat', delete=False) as f:
        fname = f.name

    try:
        io.savemat(fname, data_dict)
        loaded = io.loadmat(fname)

        for key in data_dict:
            assert key in loaded
            original_val = np.array(data_dict[key])
            loaded_val = loaded[key]

            if original_val.ndim == 0:
                original_val = np.atleast_2d(original_val)

            np.testing.assert_array_almost_equal(original_val, loaded_val)
    finally:
        if os.path.exists(fname):
            os.unlink(fname)
```

**Failing input**: `{'A': []}`

## Reproducing the Bug

```python
import tempfile
import numpy as np
from scipy import io

with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
    fname = f.name

empty_arr = np.array([])

io.savemat(fname, {'x': empty_arr}, oned_as='row')
loaded_row = io.loadmat(fname)
print(f"oned_as='row': {empty_arr.shape} -> {loaded_row['x'].shape}")
print(f"Expected: (1, 0), Got: {loaded_row['x'].shape}")

io.savemat(fname, {'x': empty_arr}, oned_as='column')
loaded_col = io.loadmat(fname)
print(f"oned_as='column': {empty_arr.shape} -> {loaded_col['x'].shape}")
print(f"Expected: (0, 1), Got: {loaded_col['x'].shape}")

non_empty = np.array([1.0, 2.0, 3.0])
io.savemat(fname, {'x': non_empty}, oned_as='row')
loaded_ne = io.loadmat(fname)
print(f"\nNon-empty with oned_as='row': {non_empty.shape} -> {loaded_ne['x'].shape}")
print(f"Expected: (1, 3), Got: {loaded_ne['x'].shape} ✓")
```

Output:
```
oned_as='row': (0,) -> (0, 0)
Expected: (1, 0), Got: (0, 0)
oned_as='column': (0,) -> (0, 0)
Expected: (0, 1), Got: (0, 0)

Non-empty with oned_as='row': (3,) -> (1, 3)
Expected: (1, 3), Got: (1, 3) ✓
```

## Why This Is A Bug

The `oned_as` parameter is documented to control how 1D NumPy arrays are written:
- `oned_as='row'`: write 1D arrays as row vectors
- `oned_as='column'`: write 1D arrays as column vectors

For a non-empty 1D array of shape (n,):
- `oned_as='row'` → shape (1, n) ✓
- `oned_as='column'` → shape (n, 1) ✓

For an empty 1D array of shape (0,):
- `oned_as='row'` → shape (0, 0) ✗ (should be (1, 0))
- `oned_as='column'` → shape (0, 0) ✗ (should be (0, 1))

This inconsistency breaks the expected behavior and violates the contract of the `oned_as` parameter. It also makes round-trip operations unpredictable for empty arrays.

## Fix

The bug likely occurs in the MATLAB file writer where empty 1D arrays are treated as a special case without checking the `oned_as` parameter. The fix would involve ensuring that empty 1D arrays follow the same logic as non-empty 1D arrays when determining their output shape based on `oned_as`.

The fix should be in the code that handles 1D array conversion, ensuring that:
```python
if oned_as == 'row':
    shape = (1, len(arr))  # even when len(arr) == 0
elif oned_as == 'column':
    shape = (len(arr), 1)  # even when len(arr) == 0
```