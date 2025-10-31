# Bug Report: pandas.core.indexers.length_of_indexer Returns Negative Length

**Target**: `pandas.core.indexers.length_of_indexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`length_of_indexer` returns negative values for certain valid slices, when it should return 0 or positive integers representing the actual length of the sliced result.

## Property-Based Test

```python
import numpy as np
from pandas.core.indexers import length_of_indexer
from hypothesis import given, strategies as st, settings

@given(
    start=st.integers(min_value=0, max_value=100),
    stop=st.integers(min_value=0, max_value=100),
    step=st.integers(min_value=1, max_value=10),
    target_len=st.integers(min_value=1, max_value=100)
)
@settings(max_examples=1000)
def test_length_of_indexer_matches_actual_length(start, stop, step, target_len):
    slc = slice(start, stop, step)
    target = np.arange(target_len)

    computed_length = length_of_indexer(slc, target)
    actual_length = len(target[slc])

    assert computed_length == actual_length
```

**Failing input**: `start=1, stop=0, step=1, target_len=1`

## Reproducing the Bug

```python
import numpy as np
from pandas.core.indexers import length_of_indexer

target = np.array([0])
slc = slice(1, 0, 1)

computed_length = length_of_indexer(slc, target)
actual_length = len(target[slc])

print(f"Computed: {computed_length}")
print(f"Actual: {actual_length}")
assert computed_length == actual_length
```

Output:
```
Computed: -1
Actual: 0
AssertionError
```

Additional failing cases:
- `slice(5, 3, 1)` returns -2 instead of 0
- `slice(10, 5, 1)` returns -5 instead of 0
- `slice(3, 5, -1)` returns -2 instead of 0

## Why This Is A Bug

The function `length_of_indexer` is documented to "Return the expected length of target[indexer]". When a slice has `start > stop` with a positive step (or `start < stop` with a negative step), Python's slicing behavior returns an empty array with length 0. However, `length_of_indexer` incorrectly returns negative values in these cases.

This violates the fundamental property that the length of any indexing operation must be non-negative.

## Fix

```diff
--- a/pandas/core/indexers/utils.py
+++ b/pandas/core/indexers/utils.py
@@ -285,7 +285,7 @@ def length_of_indexer(indexer, target=None) -> int:
         elif step < 0:
             start, stop = stop + 1, start + 1
             step = -step
-        return (stop - start + step - 1) // step
+        return max(0, (stop - start + step - 1) // step)
     elif isinstance(indexer, (ABCSeries, ABCIndex, np.ndarray, list)):
         if isinstance(indexer, list):
             indexer = np.array(indexer)
```

The fix ensures that `length_of_indexer` never returns negative values by taking the maximum of 0 and the computed length. This matches Python's slicing behavior where invalid or empty slices return empty sequences.