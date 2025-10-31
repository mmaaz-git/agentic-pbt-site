# Bug Report: pandas.core.indexers.length_of_indexer Returns Negative Values

**Target**: `pandas.core.indexers.length_of_indexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `length_of_indexer` function returns negative integers for slices that produce empty results, instead of returning 0. This violates the fundamental invariant that a length cannot be negative.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
import numpy as np
from pandas.core.indexers import length_of_indexer

@given(
    start=st.integers(min_value=-100, max_value=100) | st.none(),
    stop=st.integers(min_value=-100, max_value=100) | st.none(),
    step=st.integers(min_value=-100, max_value=100).filter(lambda x: x != 0) | st.none(),
    target_len=st.integers(min_value=0, max_value=200)
)
@settings(max_examples=500)
def test_length_of_indexer_slice(start, stop, step, target_len):
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

print(f"Computed length: {computed_length}")
print(f"Actual length: {actual_length}")
print(f"Bug: {computed_length} != {actual_length}")

test_cases = [
    (0, slice(1, None), "Empty array, start=1"),
    (0, slice(5, 10), "Empty array, start=5, stop=10"),
    (3, slice(5, None), "Array[3], start=5"),
    (5, slice(3, 2), "Array[5], start=3, stop=2"),
]

for target_len, slc, desc in test_cases:
    target = np.arange(target_len)
    computed = length_of_indexer(slc, target)
    actual = len(target[slc])
    print(f"{desc}: computed={computed}, actual={actual}, match={computed == actual}")
```

Output:
```
Computed length: -1
Actual length: 0
Bug: -1 != 0
Empty array, start=1: computed=-1, actual=0, match=False
Empty array, start=5, stop=10: computed=-5, actual=0, match=False
Array[3], start=5: computed=-2, actual=0, match=False
Array[5], start=3, stop=2: computed=-1, actual=0, match=False
```

## Why This Is A Bug

The function's docstring states it should "Return the expected length of target[indexer]". Lengths cannot be negative - this violates a fundamental invariant. When a slice produces no elements (e.g., `start >= len(target)` or `stop <= start`), the function should return 0, not a negative value.

This bug could cause issues in code that:
1. Allocates arrays based on the computed length (negative size would crash)
2. Uses the length in arithmetic without checking for negatives
3. Assumes lengths are always non-negative (a reasonable assumption)

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

The fix ensures the function always returns a non-negative integer by taking the maximum of 0 and the computed length.