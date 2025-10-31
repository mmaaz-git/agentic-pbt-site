# Bug Report: pandas.core.indexers.utils.length_of_indexer Returns Negative Length

**Target**: `pandas.core.indexers.utils.length_of_indexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `length_of_indexer` function incorrectly returns negative values when computing the length of a slice with a negative stop index on an empty or small target sequence, violating its contract to return "the expected length of target[indexer]".

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pandas.core.indexers import length_of_indexer

@given(
    target_list=st.lists(st.integers(), min_size=0, max_size=50),
    slice_start=st.integers(min_value=-60, max_value=60) | st.none(),
    slice_stop=st.integers(min_value=-60, max_value=60) | st.none(),
    slice_step=st.integers(min_value=-10, max_value=10).filter(lambda x: x != 0) | st.none(),
)
@settings(max_examples=500)
def test_length_of_indexer_slice(target_list, slice_start, slice_stop, slice_step):
    indexer = slice(slice_start, slice_stop, slice_step)
    expected_length = len(target_list[indexer])
    computed_length = length_of_indexer(indexer, target_list)
    assert computed_length == expected_length
```

**Failing input**: `target_list=[], slice_start=None, slice_stop=-1, slice_step=None`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

from pandas.core.indexers import length_of_indexer

target = []
indexer = slice(None, -1, None)

actual_length = len(target[indexer])
computed_length = length_of_indexer(indexer, target)

print(f"len(target[indexer]) = {actual_length}")
print(f"length_of_indexer(indexer, target) = {computed_length}")
```

Output:
```
len(target[indexer]) = 0
length_of_indexer(indexer, target) = -1
```

## Why This Is A Bug

The function's docstring states it returns "the expected length of target[indexer]". Since `len([])` cannot be negative (lengths are always non-negative integers), returning `-1` is incorrect. Python correctly evaluates `[][:−1]` as `[]` with length `0`.

This violates the fundamental property that for any target and indexer: `length_of_indexer(indexer, target) == len(target[indexer])`.

## Fix

The issue is in lines 309-310 of `pandas/core/indexers/utils.py`. When a negative `stop` index is converted to a positive index by adding `target_len`, the result can still be negative if `target_len` is small. The fix is to clamp the adjusted `stop` value to be at least 0:

```diff
--- a/pandas/core/indexers/utils.py
+++ b/pandas/core/indexers/utils.py
@@ -307,7 +307,7 @@ def length_of_indexer(indexer, target=None) -> int:
         if stop is None or stop > target_len:
             stop = target_len
         elif stop < 0:
-            stop += target_len
+            stop = max(0, stop + target_len)
         if step is None:
             step = 1
         elif step < 0:
```

The same issue applies to the `start` parameter:

```diff
@@ -303,7 +303,7 @@ def length_of_indexer(indexer, target=None) -> int:
         if start is None:
             start = 0
         elif start < 0:
-            start += target_len
+            start = max(0, start + target_len)
         if stop is None or stop > target_len:
             stop = target_len
```