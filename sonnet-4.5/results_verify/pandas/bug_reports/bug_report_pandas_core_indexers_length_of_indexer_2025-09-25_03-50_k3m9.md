# Bug Report: pandas.core.indexers.length_of_indexer Returns Negative Lengths

**Target**: `pandas.core.indexers.length_of_indexer`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `length_of_indexer` function returns negative lengths for certain slice inputs when it should always return non-negative integers, violating its contract of returning "the expected length of target[indexer]".

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings
from pandas.core.indexers import length_of_indexer

@given(
    target_len=st.integers(min_value=5, max_value=50),
    start=st.integers(min_value=0, max_value=49),
    stop=st.integers(min_value=0, max_value=49),
)
@settings(max_examples=1000)
def test_length_of_indexer_never_negative(target_len, start, stop):
    target = np.arange(target_len)
    indexer = slice(start, stop, 1)

    actual_length = len(target[indexer])
    predicted_length = length_of_indexer(indexer, target)

    assert predicted_length >= 0, f"length_of_indexer returned negative: {predicted_length}"
    assert predicted_length == actual_length
```

**Failing input**: `target_len=5, start=1, stop=0` (and many others)

## Reproducing the Bug

```python
import numpy as np
from pandas.core.indexers import length_of_indexer

target = np.arange(5)
indexer = slice(1, 0, 1)
actual_length = len(target[indexer])
predicted_length = length_of_indexer(indexer, target)
print(f"Actual: {actual_length}, Predicted: {predicted_length}")

target = np.arange(10)
indexer = slice(None, None, -1)
actual_length = len(target[indexer])
predicted_length = length_of_indexer(indexer, target)
print(f"Actual: {actual_length}, Predicted: {predicted_length}")
```

## Why This Is A Bug

The function's docstring states it returns "the expected length of target[indexer]". A length can never be negative - it's a fundamental invariant that len(x) >= 0 for any valid sequence. The function returns negative values in several cases:

1. When `stop < start` with positive step (e.g., `slice(1, 0, 1)` returns -1 instead of 0)
2. When negative indices resolve to invalid ranges (e.g., `slice(10, -11, 1)` returns -1 instead of 0)
3. When using negative step with None bounds (e.g., `slice(None, None, -1)` returns -10 instead of 10)
4. When start is beyond array bounds (e.g., `slice(5, 101, 1)` on length 1 returns -4 instead of 0)

## Fix

The bug occurs because the function sets default values for `start` and `stop` assuming a positive step, then later tries to transform them for negative steps. The fix is to check the step sign first and set defaults accordingly:

```diff
--- a/pandas/core/indexers/utils.py
+++ b/pandas/core/indexers/utils.py
@@ -5,18 +5,28 @@ def length_of_indexer(indexer, target=None) -> int:
     if target is not None and isinstance(indexer, slice):
         target_len = len(target)
         start = indexer.start
         stop = indexer.stop
-        step = indexer.step
-        if start is None:
-            start = 0
-        elif start < 0:
-            start += target_len
-        if stop is None or stop > target_len:
-            stop = target_len
-        elif stop < 0:
-            stop += target_len
-        if step is None:
-            step = 1
-        elif step < 0:
+        step = indexer.step if indexer.step is not None else 1
+
+        if step > 0:
+            if start is None:
+                start = 0
+            elif start < 0:
+                start += target_len
+            if stop is None:
+                stop = target_len
+            elif stop < 0:
+                stop += target_len
+            start = max(0, min(start, target_len))
+            stop = max(0, min(stop, target_len))
+        else:
+            if start is None:
+                start = target_len - 1
+            elif start < 0:
+                start += target_len
+            if stop is None:
+                stop = -1
+            elif stop < 0:
+                stop += target_len
             start, stop = stop + 1, start + 1
             step = -step
-        return (stop - start + step - 1) // step
+        return max(0, (stop - start + step - 1) // step)
```