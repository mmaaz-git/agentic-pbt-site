# Bug Report: pandas.core.indexers.length_of_indexer Negative Length

**Target**: `pandas.core.indexers.length_of_indexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `length_of_indexer` function returns negative values for empty slices where `start > stop` and `step > 0`, instead of returning 0. This causes incorrect length calculations that don't match Python's actual slice behavior.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st
from pandas.core.indexers import length_of_indexer


@given(
    st.integers(min_value=1, max_value=100),
    st.integers(min_value=-100, max_value=100).filter(lambda x: x != 0),
    st.integers(min_value=-100, max_value=100),
    st.integers(min_value=-100, max_value=100),
)
def test_length_of_indexer_slice_matches_actual_length(target_len, step, start, stop):
    target = list(range(target_len))
    slc = slice(start, stop, step)

    calculated_length = length_of_indexer(slc, target)
    actual_slice = target[slc]
    actual_length = len(actual_slice)

    assert calculated_length == actual_length, \
        f"length_of_indexer({slc}, target={target_len}) = {calculated_length}, but len(target[{slc}]) = {actual_length}"
```

**Failing input**: `target_len=1, step=1, start=1, stop=0`

## Reproducing the Bug

```python
from pandas.core.indexers import length_of_indexer

target = [0]
slc = slice(1, 0, 1)

calculated_length = length_of_indexer(slc, target)
actual_length = len(target[slc])

print(f"Calculated: {calculated_length}, Actual: {actual_length}")

assert calculated_length == actual_length
```

Output:
```
Calculated: -1, Actual: 0
AssertionError
```

Other failing examples:
- `slice(2, 1, 1)` returns -1 instead of 0
- `slice(4, 2, 1)` returns -2 instead of 0
- `slice(None, None, -1)` returns -5 instead of 5 (for target of length 5)
- `slice(2, 4, -1)` returns -2 instead of 0

## Why This Is A Bug

The `length_of_indexer` function claims to "Return the expected length of target[indexer]". However, it returns negative lengths for valid slices that Python evaluates to empty sequences. This violates the fundamental invariant that a length must be non-negative.

The bug occurs because the formula `(stop - start + step - 1) // step` can produce negative results when `start > stop` after normalization for negative steps. Python's slice behavior always produces length 0 for such cases, never negative lengths.

This affects any pandas code that uses `length_of_indexer` to pre-allocate arrays or validate lengths, potentially causing downstream errors or incorrect behavior.

## Fix

```diff
--- a/pandas/core/indexers/utils.py
+++ b/pandas/core/indexers/utils.py
@@ -40,7 +40,7 @@ def length_of_indexer(indexer, target=None) -> int:
         elif step < 0:
             start, stop = stop + 1, start + 1
             step = -step
-        return (stop - start + step - 1) // step
+        return max(0, (stop - start + step - 1) // step)
     elif isinstance(indexer, (ABCSeries, ABCIndex, np.ndarray, list)):
         if isinstance(indexer, list):
             indexer = np.array(indexer)
```