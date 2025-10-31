# Bug Report: scipy.io.matlab Empty 1D Array Round-trip Shape Inconsistency

**Target**: `scipy.io.matlab.savemat` and `scipy.io.matlab.loadmat`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When saving and loading empty 1D NumPy arrays via `savemat`/`loadmat`, the shape is inconsistently transformed compared to non-empty 1D arrays. Empty arrays become `(0, 0)` instead of the expected `(1, 0)` for `oned_as='row'` or `(0, 1)` for `oned_as='column'`.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/scipy_env')

import numpy as np
import scipy.io.matlab as sio
from hypothesis import given, strategies as st, settings
from io import BytesIO


@settings(max_examples=200)
@given(st.integers(min_value=0, max_value=1000))
def test_roundtrip_1d_int_array(size):
    data = np.arange(size, dtype=np.int32)
    mdict = {'data': data}

    f = BytesIO()
    sio.savemat(f, mdict)
    f.seek(0)

    loaded = sio.loadmat(f)

    assert 'data' in loaded
    np.testing.assert_array_equal(loaded['data'].squeeze(), data)
```

**Failing input**: `size=0`

## Reproducing the Bug

```python
import numpy as np
import scipy.io.matlab as sio
from io import BytesIO

data_empty = np.array([], dtype=np.int32)
data_nonempty = np.array([1], dtype=np.int32)

f = BytesIO()
sio.savemat(f, {'empty': data_empty, 'nonempty': data_nonempty}, oned_as='row')
f.seek(0)
loaded = sio.loadmat(f)

print("oned_as='row':")
print(f"  Non-empty (1,) -> {loaded['nonempty'].shape}")
print(f"  Empty (0,)     -> {loaded['empty'].shape}")

f = BytesIO()
sio.savemat(f, {'empty': data_empty, 'nonempty': data_nonempty}, oned_as='column')
f.seek(0)
loaded = sio.loadmat(f)

print("\noned_as='column':")
print(f"  Non-empty (1,) -> {loaded['nonempty'].shape}")
print(f"  Empty (0,)     -> {loaded['empty'].shape}")
```

**Output:**
```
oned_as='row':
  Non-empty (1,) -> (1, 1)
  Empty (0,)     -> (0, 0)

oned_as='column':
  Non-empty (1,) -> (1, 1)
  Empty (0,)     -> (0, 0)
```

## Why This Is A Bug

The `oned_as` parameter in `savemat` controls how 1D arrays are written to MATLAB files:
- `oned_as='row'`: writes 1D arrays as row vectors (shape becomes `(1, n)`)
- `oned_as='column'`: writes 1D arrays as column vectors (shape becomes `(n, 1)`)

For **non-empty** 1D arrays, the behavior is correct and consistent:
- `oned_as='row'`: `(n,)` → `(1, n)`  ✓
- `oned_as='column'`: `(n,)` → `(n, 1)`  ✓

For **empty** 1D arrays, the behavior is **inconsistent**:
- `oned_as='row'`: `(0,)` → `(0, 0)` ✗ (should be `(1, 0)`)
- `oned_as='column'`: `(0,)` → `(0, 0)` ✗ (should be `(0, 1)`)

This violates the fundamental invariant that empty arrays should follow the same transformation rules as non-empty arrays. The inconsistency can cause unexpected behavior in code that processes arrays of varying sizes.

## Fix

The issue likely stems from special-case handling of empty arrays that doesn't respect the `oned_as` parameter. The fix would involve ensuring that when a 1D array of shape `(0,)` is written:
- With `oned_as='row'`, it should be written as shape `(1, 0)`
- With `oned_as='column'`, it should be written as shape `(0, 1)`

This would require modifying the array writing logic in the MATLAB file writer to apply the `oned_as` transformation consistently regardless of whether the array is empty or non-empty.