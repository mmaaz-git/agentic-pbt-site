# Bug Report: pandas.core.indexers.length_of_indexer Negative Step Slice

**Target**: `pandas.core.indexers.length_of_indexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `length_of_indexer` function returns incorrect (negative) values for slices with negative steps and None start/stop values, such as `slice(None, None, -1)` (equivalent to `[::-1]`).

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import numpy as np
from hypothesis import given, strategies as st
from pandas.core.indexers import length_of_indexer


@given(
    st.integers(min_value=1, max_value=1000),
    st.integers(min_value=-100, max_value=100) | st.none(),
    st.integers(min_value=-100, max_value=100) | st.none(),
    st.integers(min_value=-10, max_value=10).filter(lambda x: x != 0) | st.none(),
)
def test_length_of_indexer_matches_actual_length(target_len, start, stop, step):
    target = list(range(target_len))
    indexer = slice(start, stop, step)

    expected_length = length_of_indexer(indexer, target)
    actual_length = len(target[indexer])

    assert expected_length == actual_length
```

**Failing input**: `slice(None, None, -1)` with any non-empty target

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

from pandas.core.indexers import length_of_indexer

target = [0]
indexer = slice(None, None, -1)

result = length_of_indexer(indexer, target)
expected = len(target[indexer])

print(f"length_of_indexer result: {result}")
print(f"Expected: {expected}")

target2 = [0, 1, 2, 3, 4]
result2 = length_of_indexer(slice(None, None, -1), target2)
expected2 = len(target2[::-1])

print(f"length_of_indexer([0,1,2,3,4], [::-1]): {result2}")
print(f"Expected: {expected2}")
```

Output:
```
length_of_indexer result: -1
Expected: 1
length_of_indexer([0,1,2,3,4], [::-1]): -5
Expected: 5
```

## Why This Is A Bug

The function incorrectly computes the length of slices with negative steps. In Python, `slice(None, None, -1)` (i.e., `[::-1]`) reverses the entire sequence. For a list of length n, this should always return n elements, but `length_of_indexer` returns -n instead.

The root cause is in lines 303-315 of `/pandas/core/indexers/utils.py`:

```python
if start is None:
    start = 0  # Wrong for negative step!
elif start < 0:
    start += target_len
if stop is None or stop > target_len:
    stop = target_len  # Wrong for negative step!
elif stop < 0:
    stop += target_len
if step is None:
    step = 1
elif step < 0:
    start, stop = stop + 1, start + 1
    step = -step
```

The function normalizes `start=None` to 0 and `stop=None` to `target_len`, which is correct for positive steps. However, for negative steps in Python:
- `start=None` should be interpreted as `target_len - 1` (start from the end)
- `stop=None` should be interpreted as before index 0 (go all the way to the beginning)

## Fix

The function needs to check the step value before normalizing None values:

```diff
--- a/pandas/core/indexers/utils.py
+++ b/pandas/core/indexers/utils.py
@@ -298,22 +298,27 @@ def length_of_indexer(indexer, target=None) -> int:
     if target is not None and isinstance(indexer, slice):
         target_len = len(target)
         start = indexer.start
         stop = indexer.stop
         step = indexer.step
+        if step is None:
+            step = 1
+
+        # Handle None values based on step direction
+        if step < 0:
+            # For negative step, None has different semantics
+            if start is None:
+                start = target_len - 1
+            elif start < 0:
+                start += target_len
+            if stop is None:
+                stop = -1
+            elif stop < 0:
+                stop += target_len
+        else:
+            # For positive step
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
+            if start is None:
+                start = 0
+            elif start < 0:
+                start += target_len
+            if stop is None or stop > target_len:
+                stop = target_len
+            elif stop < 0:
+                stop += target_len
+
+        if step < 0:
             start, stop = stop + 1, start + 1
             step = -step
         return (stop - start + step - 1) // step
```