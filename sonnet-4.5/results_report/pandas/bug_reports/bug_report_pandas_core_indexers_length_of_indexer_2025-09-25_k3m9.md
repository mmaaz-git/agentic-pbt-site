# Bug Report: pandas.core.indexers.length_of_indexer Multiple Logic Errors

**Target**: `pandas.core.indexers.length_of_indexer`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `length_of_indexer` function has multiple logic errors that cause it to return incorrect (often negative) values for various edge cases, violating its documented contract of returning the "expected length of target[indexer]". The function returns negative lengths and incorrect values for:
1. Slices with negative steps
2. Range objects with large steps
3. Slices where start > len(target)
4. Slices where stop is negative and out of bounds

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
def test_length_of_indexer_matches_actual(n, start, stop, step):
    target = np.arange(n)
    indexer = slice(start, stop, step)

    expected_len = length_of_indexer(indexer, target)
    actual_len = len(target[indexer])

    assert expected_len == actual_len
```

**Failing inputs**:
- Bug 1: `n=1, start=None, stop=None, step=-1`
- Bug 2: `range(0, 1, 2)`
- Bug 3: `n=1, start=2, stop=None, step=None`
- Bug 4: `n=1, start=None, stop=-5, step=None`

## Reproducing the Bugs

### Bug 1: Negative step slices return negative lengths

```python
import numpy as np
from pandas.core.indexers import length_of_indexer

target = np.array([0])
result = length_of_indexer(slice(None, None, -1), target)
actual = len(target[slice(None, None, -1)])

print(f"Result: {result}, Actual: {actual}")
```

Output: `Result: -1, Actual: 1`

### Bug 2: Range with step > (stop - start)

```python
from pandas.core.indexers import length_of_indexer

r = range(0, 1, 2)
result = length_of_indexer(r, None)
actual = len(r)

print(f"Result: {result}, Actual: {actual}")
```

Output: `Result: 0, Actual: 1`

### Bug 3: Slice with start > len(target)

```python
import numpy as np
from pandas.core.indexers import length_of_indexer

target = np.array([0])
result = length_of_indexer(slice(2, None), target)
actual = len(target[slice(2, None)])

print(f"Result: {result}, Actual: {actual}")
```

Output: `Result: -1, Actual: 0`

### Bug 4: Slice with negative stop out of bounds

```python
import numpy as np
from pandas.core.indexers import length_of_indexer

target = np.array([0])
result = length_of_indexer(slice(None, -5), target)
actual = len(target[slice(None, -5)])

print(f"Result: {result}, Actual: {actual}")
```

Output: `Result: -4, Actual: 0`

## Why This Is A Bug

The function's docstring (line 292) states it should "Return the expected length of target[indexer]". Lengths must always be non-negative. The function returns negative values for multiple edge cases, which violates basic expectations and makes the function unusable for its intended purpose.

## Root Causes

1. **Bug 1**: Lines 303-310 set default values for None assuming a positive step. For negative steps, start should default to `target_len - 1` and stop to `-target_len - 1`.

2. **Bug 2**: Line 326 uses floor division `(stop - start) // step` which is incorrect. Should use ceiling division: `(stop - start + step - 1) // step` (only for positive steps).

3. **Bug 3**: After normalizing start (line 306), the code doesn't check if `start >= target_len`, leading to negative results.

4. **Bug 4**: Line 310 adds target_len to negative stop values, but doesn't handle the case where the result is still negative (completely out of bounds).

## Fix

```diff
--- a/pandas/core/indexers/utils.py
+++ b/pandas/core/indexers/utils.py
@@ -298,25 +298,35 @@ def length_of_indexer(indexer, target=None) -> int:
     if target is not None and isinstance(indexer, slice):
         target_len = len(target)
         start = indexer.start
         stop = indexer.stop
         step = indexer.step
+
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
+
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
+
+        if step < 0:
             start, stop = stop + 1, start + 1
             step = -step
-        return (stop - start + step - 1) // step
+
+        return max(0, (stop - start + step - 1) // step)
     elif isinstance(indexer, (ABCSeries, ABCIndex, np.ndarray, list)):
         if isinstance(indexer, list):
             indexer = np.array(indexer)
@@ -324,7 +334,7 @@ def length_of_indexer(indexer, target=None) -> int:
             return indexer.sum()
         return len(indexer)
     elif isinstance(indexer, range):
-        return (indexer.stop - indexer.start) // indexer.step
+        return max(0, (indexer.stop - indexer.start + indexer.step - 1) // indexer.step)
     elif not is_list_like_indexer(indexer):
         return 1
     raise AssertionError("cannot find the length of the indexer")
```