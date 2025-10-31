# Bug Report: pandas.core.indexers.length_of_indexer Negative Step

**Target**: `pandas.core.indexers.length_of_indexer`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `length_of_indexer` function incorrectly returns negative values for slices with negative steps, violating its documented contract of returning the "expected length of target[indexer]".

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
from pandas.core.indexers import length_of_indexer

@given(
    n=st.integers(min_value=1, max_value=1000),
    start=st.integers(min_value=-100, max_value=100) | st.none(),
    stop=st.integers(min_value=-100, max_value=100) | st.none(),
    step=st.integers(min_value=-10, max_value=10).filter(lambda x: x != 0) | st.none(),
)
@settings(max_examples=500)
def test_length_of_indexer_slice_matches_actual_length(n, start, stop, step):
    target = np.arange(n)
    indexer = slice(start, stop, step)

    expected_len = length_of_indexer(indexer, target)
    actual_len = len(target[indexer])

    assert expected_len == actual_len
```

**Failing input**: `n=1, start=None, stop=None, step=-1`

## Reproducing the Bug

```python
import numpy as np
from pandas.core.indexers import length_of_indexer

target = np.array([0])
slc = slice(None, None, -1)

result = length_of_indexer(slc, target)
actual = len(target[slc])

print(f"length_of_indexer: {result}")
print(f"Actual length: {actual}")

assert result == actual
```

Output:
```
length_of_indexer: -1
Actual length: 1
AssertionError
```

Additional failing cases:
- `slice(None, None, -1)` on array of length 5: returns -5, should be 5
- `slice(None, None, -2)` on array of length 10: returns -5, should be 5
- `slice(5, None, -1)` on array of length 10: returns -5, should be 6

## Why This Is A Bug

The function's docstring (line 292) states it should "Return the expected length of target[indexer]". Length must always be non-negative. The function returns negative values for slices with negative steps, which violates basic expectations and makes the function unusable for its intended purpose.

## Fix

The bug occurs because lines 303-310 set default values for None assuming a positive step. For negative steps, the defaults should be different:
- When `step < 0` and `start is None`: start should default to `target_len - 1` (last index)
- When `step < 0` and `stop is None`: stop should default to `-target_len - 1` (before first index)

```diff
--- a/pandas/core/indexers/utils.py
+++ b/pandas/core/indexers/utils.py
@@ -301,16 +301,23 @@ def length_of_indexer(indexer, target=None) -> int:
         stop = indexer.stop
         step = indexer.step
+        if step is None:
+            step = 1
+
         if start is None:
-            start = 0
+            if step < 0:
+                start = target_len - 1
+            else:
+                start = 0
         elif start < 0:
             start += target_len
-        if stop is None or stop > target_len:
-            stop = target_len
+        if stop is None:
+            if step < 0:
+                stop = -target_len - 1
+            else:
+                stop = target_len
+        elif stop > target_len:
+            stop = target_len
         elif stop < 0:
             stop += target_len
-        if step is None:
-            step = 1
-        elif step < 0:
+        if step < 0:
             start, stop = stop + 1, start + 1
             step = -step
         return (stop - start + step - 1) // step
```