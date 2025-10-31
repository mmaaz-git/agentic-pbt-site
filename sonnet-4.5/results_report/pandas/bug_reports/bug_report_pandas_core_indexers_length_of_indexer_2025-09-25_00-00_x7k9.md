# Bug Report: pandas.core.indexers length_of_indexer Returns Negative Length

**Target**: `pandas.core.indexers.length_of_indexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`length_of_indexer` returns `-1` for empty slices where `start > stop` (with positive step), but these slices produce 0 elements when actually used.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
from pandas.core.indexers import length_of_indexer

@given(
    start=st.none() | st.integers(min_value=-50, max_value=50),
    stop=st.none() | st.integers(min_value=-50, max_value=50),
    step=st.none() | st.integers(min_value=-10, max_value=10).filter(lambda x: x != 0),
)
@settings(max_examples=500)
def test_length_of_indexer_matches_actual_slice(start, stop, step):
    target_len = 50
    target = np.arange(target_len)
    slc = slice(start, stop, step)
    expected_len = length_of_indexer(slc, target)
    actual_sliced = target[slc]
    actual_len = len(actual_sliced)
    assert expected_len == actual_len
```

**Failing inputs**:
- `slice(1, 0, None)` (positive step, start > stop)
- `slice(0, 1, -1)` (negative step, start < stop)

## Reproducing the Bug

```python
import numpy as np
from pandas.core.indexers import length_of_indexer

target = np.arange(50)

slc1 = slice(1, 0, None)
print(f"slice(1, 0, None):")
print(f"  length_of_indexer: {length_of_indexer(slc1, target)}")
print(f"  Actual length: {len(target[slc1])}")

slc2 = slice(0, 1, -1)
print(f"\nslice(0, 1, -1):")
print(f"  length_of_indexer: {length_of_indexer(slc2, target)}")
print(f"  Actual length: {len(target[slc2])}")
```

Output:
```
slice(1, 0, None):
  length_of_indexer: -1
  Actual length: 0

slice(0, 1, -1):
  length_of_indexer: -1
  Actual length: 0
```

## Why This Is A Bug

When `start > stop` with a positive step (or `start < stop` with a negative step), Python slicing produces an empty sequence with length 0. The function incorrectly returns -1 in this case.

The docstring on line 292 states: "Return the expected length of target[indexer]". A negative length is never valid for a slicing operation.

## Fix

```diff
--- a/pandas/core/indexers/utils.py
+++ b/pandas/core/indexers/utils.py
@@ -313,6 +313,10 @@ def length_of_indexer(indexer, target=None) -> int:
         elif step < 0:
             start, stop = stop + 1, start + 1
             step = -step
+
+        if start >= stop:
+            return 0
+
         return (stop - start + step - 1) // step
     elif isinstance(indexer, (ABCSeries, ABCIndex, np.ndarray, list)):
         if isinstance(indexer, list):
```