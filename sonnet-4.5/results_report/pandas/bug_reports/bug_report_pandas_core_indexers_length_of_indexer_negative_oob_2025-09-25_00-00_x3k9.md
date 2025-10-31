# Bug Report: pandas.core.indexers.length_of_indexer Negative Out-of-Bounds Slice

**Target**: `pandas.core.indexers.length_of_indexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `length_of_indexer` function returns negative lengths for slices with out-of-bounds negative indices, violating the fundamental property that `length_of_indexer(indexer, target) == len(target[indexer])`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
from pandas.core.indexers import length_of_indexer

@given(
    target=st.lists(st.integers(), min_size=1, max_size=50),
    slice_start=st.one_of(st.none(), st.integers(min_value=-50, max_value=50)),
    slice_stop=st.one_of(st.none(), st.integers(min_value=-50, max_value=50)),
    slice_step=st.one_of(st.none(), st.integers(min_value=1, max_value=10))
)
def test_length_of_indexer_slice_consistency(target, slice_start, slice_stop, slice_step):
    target_array = np.array(target)
    indexer = slice(slice_start, slice_stop, slice_step)

    actual_length = len(target_array[indexer])
    predicted_length = length_of_indexer(indexer, target_array)

    assert actual_length == predicted_length
```

**Failing inputs**:
- `slice(None, -2, None)` on array of length 1 (stop out of bounds)
- `slice(2, None)` on array of length 1 (start out of bounds)

## Reproducing the Bug

```python
import numpy as np
from pandas.core.indexers import length_of_indexer

target = np.array([0])

indexer1 = slice(None, -2, None)
print(f"Case 1: {indexer1}")
print(f"  Actual: {len(target[indexer1])}, Predicted: {length_of_indexer(indexer1, target)}")

indexer2 = slice(2, None)
print(f"Case 2: {indexer2}")
print(f"  Actual: {len(target[indexer2])}, Predicted: {length_of_indexer(indexer2, target)}")
```

## Why This Is A Bug

The function's docstring states "Return the expected length of target[indexer]", but it returns -1 instead of 0 for slices where indices go out of bounds.

The bug occurs in lines 303-310 of `utils.py`:

```python
if start is None:
    start = 0
elif start < 0:
    start += target_len
if stop is None or stop > target_len:
    stop = target_len
elif stop < 0:
    stop += target_len
```

For `slice(None, -2, None)` with `target_len=1`:
- `stop = -2`
- After `stop += target_len`: `stop = -1` (still negative, not clamped to 0)
- Final calculation: `(-1 - 0 + 1 - 1) // 1 = -1`

For `slice(2, None)` with `target_len=1`:
- `start = 2` (not clamped, should be clamped to target_len)
- `stop = 1`
- Final calculation: `(1 - 2 + 1 - 1) // 1 = -1`

## Fix

```diff
--- a/pandas/core/indexers/utils.py
+++ b/pandas/core/indexers/utils.py
@@ -303,11 +303,17 @@ def length_of_indexer(indexer, target=None) -> int:
         if start is None:
             start = 0
         elif start < 0:
             start += target_len
+            if start < 0:
+                start = 0
+        if start > target_len:
+            start = target_len
         if stop is None or stop > target_len:
             stop = target_len
         elif stop < 0:
             stop += target_len
+            if stop < 0:
+                stop = 0
         if step is None:
             step = 1
         elif step < 0:
             start, stop = stop + 1, start + 1
             step = -step
         return (stop - start + step - 1) // step
```