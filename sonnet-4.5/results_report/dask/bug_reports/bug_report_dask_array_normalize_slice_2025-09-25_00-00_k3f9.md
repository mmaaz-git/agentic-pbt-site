# Bug Report: dask.array.slicing normalize_slice Empty Slice Mishandling

**Target**: `dask.array.slicing.normalize_slice`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`normalize_slice` incorrectly normalizes negative-indexed slices with negative step when start equals stop, producing a non-empty slice instead of an empty one. This causes dask.array to return different results than NumPy for certain edge case slices.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

import numpy as np
from hypothesis import assume, given, settings, strategies as st
from dask.array.slicing import normalize_slice


@given(
    st.integers(min_value=-20, max_value=20),
    st.integers(min_value=-20, max_value=20),
    st.integers(min_value=-10, max_value=10).filter(lambda x: x != 0),
    st.integers(min_value=1, max_value=100)
)
@settings(max_examples=500)
def test_normalize_slice_equivalence(start, stop, step, dim):
    assume(dim > 0)
    arr = np.arange(dim)
    idx = slice(start, stop, step)

    normalized = normalize_slice(idx, dim)

    original_result = arr[idx]
    normalized_result = arr[normalized]

    assert np.array_equal(original_result, normalized_result), \
        f"Normalized slice should produce same elements: {idx} vs {normalized} on array of length {dim}"
```

**Failing input**: `start=-2, stop=-2, step=-1, dim=1`

## Reproducing the Bug

```python
import numpy as np
import dask.array as da

arr = np.array([0])
darr = da.from_array(arr, chunks=-1)

idx = slice(-2, -2, -1)

np_result = arr[idx]
dask_result = darr[idx].compute()

print(f"NumPy: arr[{idx}] = {np_result}")
print(f"Dask:  darr[{idx}].compute() = {dask_result}")

assert np.array_equal(np_result, dask_result)
```

Output:
```
NumPy: arr[slice(-2, -2, -1)] = []
Dask:  darr[slice(-2, -2, -1)].compute() = [0]
AssertionError
```

## Why This Is A Bug

When `slice(-2, -2, -1)` is applied to a 1D array:
- NumPy correctly interprets this as an empty slice (start equals stop with negative step)
- `normalize_slice` calls `idx.indices(1)` which returns `start=-1, stop=-1, step=-1`
- The function then normalizes by setting `stop=None` when `stop < 0` for negative steps
- This changes `slice(-1, -1, -1)` to `slice(-1, None, -1)`, which is NOT equivalent
- `arr[slice(-1, None, -1)]` returns the entire array reversed, not an empty array

The bug is in the normalization logic for negative steps at lines 791-795 of `/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/array/slicing.py`:

```python
elif step < 0:
    if start >= dim - 1:
        start = None
    if stop < 0:
        stop = None
```

This logic doesn't account for the case where `start == stop` with negative step, which should remain an empty slice.

## Fix

```diff
--- a/dask/array/slicing.py
+++ b/dask/array/slicing.py
@@ -789,6 +789,10 @@ def normalize_slice(idx, dim):
             if stop is not None and start is not None and stop < start:
                 stop = start
         elif step < 0:
+            # Preserve empty slice when start == stop
+            if start == stop:
+                if start >= dim - 1:
+                    return slice(None, 0, step)
+                return slice(start, start, step)
             if start >= dim - 1:
                 start = None
             if stop < 0:
```