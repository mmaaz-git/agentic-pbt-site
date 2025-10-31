# Bug Report: pandas.core.indexers.length_of_indexer Negative Length for Out-of-Bounds Slice

**Target**: `pandas.core.indexers.length_of_indexer`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `length_of_indexer` function returns negative lengths for slices with out-of-bounds start indices, which is nonsensical. Lengths must be non-negative.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
from pandas.core.indexers import length_of_indexer

@given(
    target=st.lists(st.integers(), min_size=1, max_size=100),
    slice_start=st.one_of(st.none(), st.integers(min_value=-50, max_value=50)),
    slice_stop=st.one_of(st.none(), st.integers(min_value=-50, max_value=50)),
    slice_step=st.one_of(st.none(), st.integers(min_value=1, max_value=5))
)
@settings(max_examples=500)
def test_length_of_indexer_slice_positive_step_consistency(target, slice_start, slice_stop, slice_step):
    target_array = np.array(target)
    indexer = slice(slice_start, slice_stop, slice_step)

    actual_length = len(target_array[indexer])
    predicted_length = length_of_indexer(indexer, target_array)

    assert actual_length == predicted_length
```

**Failing input**: `slice(2, None, None)` on array `[0]` (length 1)

## Reproducing the Bug

```python
import numpy as np
from pandas.core.indexers import length_of_indexer

array = np.array([0])
indexer = slice(2, None, None)

actual_result = array[indexer]
actual_length = len(actual_result)
predicted_length = length_of_indexer(indexer, array)

print(f"Array: {array}")
print(f"Indexer: {indexer}")
print(f"Actual result: {actual_result}")
print(f"Actual length: {actual_length}")
print(f"Predicted length: {predicted_length}")
print(f"Bug: {predicted_length < 0}")
```

Output:
```
Array: [0]
Indexer: slice(2, None, None)
Actual result: []
Actual length: 0
Predicted length: -1
Bug: True
```

## Why This Is A Bug

Returning a negative length is nonsensical and can cause downstream errors. When `start >= stop` after normalization, the slice produces an empty result, so the length should be 0.

The bug occurs in `utils.py` line 316:

```python
return (stop - start + step - 1) // step
```

For `slice(2, None)` on array of length 1:
- After normalization: start=2, stop=1, step=1
- Formula: `(1 - 2 + 1 - 1) // 1 = -1`
- Correct: `max(0, (1 - 2 + 1 - 1) // 1) = 0`

## Fix

```diff
--- a/pandas/core/indexers/utils.py
+++ b/pandas/core/indexers/utils.py
@@ -313,7 +313,7 @@ def length_of_indexer(indexer, target=None) -> int:
         elif step < 0:
             start, stop = stop + 1, start + 1
             step = -step
-        return (stop - start + step - 1) // step
+        return max(0, (stop - start + step - 1) // step)
     elif isinstance(indexer, (ABCSeries, ABCIndex, np.ndarray, list)):
         if isinstance(indexer, list):
             indexer = np.array(indexer)
```