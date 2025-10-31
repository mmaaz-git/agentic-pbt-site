# Bug Report: pandas.core.indexers.length_of_indexer Returns Negative Length

**Target**: `pandas.core.indexers.length_of_indexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `length_of_indexer()` function incorrectly returns negative values when computing the length of a slice where `start >= len(target)`. In such cases, the actual slice produces an empty result (length 0), but the function returns negative numbers.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
from pandas.core.indexers import length_of_indexer


@given(
    start=st.integers(min_value=0, max_value=100) | st.none(),
    stop=st.integers(min_value=0, max_value=100) | st.none(),
    step=st.integers(min_value=1, max_value=10) | st.none(),
    target_len=st.integers(min_value=0, max_value=100),
)
def test_length_of_indexer_slice_matches_actual(start, stop, step, target_len):
    slc = slice(start, stop, step)
    target = np.arange(target_len)

    computed_length = length_of_indexer(slc, target)
    actual_length = len(target[slc])

    assert computed_length == actual_length
```

**Failing input**: `start=1, stop=None, step=None, target_len=0`

## Reproducing the Bug

```python
import numpy as np
from pandas.core.indexers import length_of_indexer

target = np.arange(0)
slc = slice(1, None, None)
computed_length = length_of_indexer(slc, target)
actual_length = len(target[slc])
print(f"Computed: {computed_length}, Actual: {actual_length}")

target = np.arange(5)
slc = slice(10, None, None)
computed_length = length_of_indexer(slc, target)
actual_length = len(target[slc])
print(f"Computed: {computed_length}, Actual: {actual_length}")
```

## Why This Is A Bug

When a slice's `start` index is beyond the array length, NumPy correctly returns an empty array (length 0). However, `length_of_indexer()` returns negative values in these cases. This violates the fundamental invariant that `length_of_indexer(slc, target)` should equal `len(target[slc])` for all valid slices.

The bug occurs in the formula `(stop - start + step - 1) // step` when `start > stop`. For example, with `start=1, stop=0, step=1`, this becomes `(0 - 1 + 1 - 1) // 1 = -1`.

This bug has real-world impact: `check_setitem_lengths()` uses `length_of_indexer()` to validate assignments. When trying to assign an empty list to an out-of-bounds slice (e.g., `arr[10:] = []` where `len(arr) == 5`), the function incorrectly raises ValueError even though this is a valid no-op assignment.

```python
import numpy as np
from pandas.core.indexers import check_setitem_lengths

values = np.array([1, 2, 3, 4, 5])
indexer = slice(10, None)
value = []
check_setitem_lengths(indexer, value, values)
```

## Fix

```diff
--- a/pandas/core/indexers/utils.py
+++ b/pandas/core/indexers/utils.py
@@ -313,7 +313,10 @@ def length_of_indexer(indexer, target=None) -> int:
     elif step < 0:
         start, stop = stop + 1, start + 1
         step = -step
-    return (stop - start + step - 1) // step
+    if start >= stop:
+        return 0
+    else:
+        return (stop - start + step - 1) // step
 elif isinstance(indexer, (ABCSeries, ABCIndex, np.ndarray, list)):
     if isinstance(indexer, list):
         indexer = np.array(indexer)
```