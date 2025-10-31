# Bug Report: numpy.ma clump_masked/clump_unmasked IndexError on Empty Arrays

**Target**: `numpy.ma.clump_masked`, `numpy.ma.clump_unmasked` (via `numpy.ma.extras._ezclump`)
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`clump_masked()` and `clump_unmasked()` crash with IndexError when given empty masked arrays because the underlying `_ezclump()` function attempts to access `mask[0]` without checking if the mask is empty.

## Property-Based Test

```python
import numpy as np
import numpy.ma as ma
from hypothesis import given, strategies as st, settings
from hypothesis.extra import numpy as npst

@st.composite
def masked_arrays_1d(draw, dtype=np.int64, max_size=50):
    size = draw(st.integers(min_value=0, max_value=max_size))
    data = draw(npst.arrays(dtype=dtype, shape=(size,),
                           elements=st.integers(min_value=-1000, max_value=1000)))
    mask = draw(npst.arrays(dtype=bool, shape=(size,)))
    return ma.array(data, mask=mask)

@given(masked_arrays_1d())
@settings(max_examples=500)
def test_clump_masked_partition(arr):
    clumps = ma.clump_masked(arr)
    mask = ma.getmaskarray(arr)
    covered_indices = set()
    for clump in clumps:
        for i in range(clump.start, clump.stop):
            assert mask[i]
            covered_indices.add(i)
    for i in range(len(arr)):
        if mask[i]:
            assert i in covered_indices
```

**Failing input**: `masked_array(data=[], mask=[], dtype=int64)`

## Reproducing the Bug

```python
import numpy as np
import numpy.ma as ma

empty_arr = ma.array([], dtype=int, mask=[])
result = ma.clump_masked(empty_arr)
```

Output:
```
IndexError: index 0 is out of bounds for axis 0 with size 0
```

Same crash occurs with `clump_unmasked()`:

```python
result = ma.clump_unmasked(empty_arr)
```

## Why This Is A Bug

Empty arrays are valid inputs to masked array functions. The clumping functions should handle them gracefully by returning an empty list of slices, not crashing with IndexError.

The root cause is in `_ezclump()` at line 2199 of `numpy/ma/extras.py`, which accesses `mask[0]` without first checking if the mask has size 0.

## Fix

```diff
--- a/numpy/ma/extras.py
+++ b/numpy/ma/extras.py
@@ -2194,6 +2194,9 @@ def _ezclump(mask):
     if mask.ndim > 1:
         mask = mask.ravel()
     idx = (mask[1:] ^ mask[:-1]).nonzero()
     idx = idx[0] + 1

+    if mask.size == 0:
+        return []
+
     if mask[0]:
         if len(idx) == 0:
             return [slice(0, mask.size)]
```