# Bug Report: LRU Cache Evicts Recently Accessed Items

**Target**: `dask.dataframe.dask_expr._util.LRU`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `LRU` class incorrectly evicts items that have been recently accessed when the cache is at maximum capacity. According to the class docstring, it should evict "the least recently looked-up key when full", but it fails to do so when accessing an item causes it to be moved to the end of the cache.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from dask.dataframe.dask_expr._util import LRU

@given(st.integers(min_value=1, max_value=10))
def test_lru_get_moves_to_end(maxsize):
    lru = LRU(maxsize)
    for i in range(maxsize):
        lru[i] = i

    first_key = 0
    _ = lru[first_key]

    lru[maxsize] = maxsize

    assert first_key in lru
```

**Failing input**: `maxsize=1`

## Reproducing the Bug

```python
from dask.dataframe.dask_expr._util import LRU

lru = LRU(maxsize=1)

lru[0] = "value0"

lru[0]

lru[1] = "value1"

assert 0 in lru, "Bug: Key 0 was evicted even though it was accessed more recently"
```

## Why This Is A Bug

The `LRU` class documentation states it is a "Limited size mapping, evicting the least recently looked-up key when full". However, when the cache is at maximum capacity:

1. Adding key 0 with value "value0" fills the cache (size=1)
2. Accessing key 0 with `lru[0]` calls `__getitem__`, which moves key 0 to the end (most recently used position)
3. Adding key 1 should evict the **least** recently looked-up key, but instead evicts key 0, which was just accessed

This violates the LRU (Least Recently Used) semantics promised by the class.

## Root Cause

In `__setitem__` (line 109-112 of `_util.py`):

```python
def __setitem__(self, key: K, value: V) -> None:
    if len(self) >= self.maxsize:
        cast(OrderedDict, self.data).popitem(last=False)
    super().__setitem__(key, value)
```

The bug is that the code unconditionally evicts an item when `len(self) >= self.maxsize`, without checking if the key already exists in the cache. When updating an existing key, the cache size doesn't increase, so no eviction should occur.

## Fix

```diff
--- a/dask/dataframe/dask_expr/_util.py
+++ b/dask/dataframe/dask_expr/_util.py
@@ -108,6 +108,9 @@ class LRU(UserDict[K, V]):

     def __setitem__(self, key: K, value: V) -> None:
-        if len(self) >= self.maxsize:
+        if key in self.data:
+            cast(OrderedDict, self.data).move_to_end(key)
+        elif len(self) >= self.maxsize:
             cast(OrderedDict, self.data).popitem(last=False)
         super().__setitem__(key, value)
```

This fix:
1. Checks if the key already exists in the cache
2. If it does, moves it to the end (most recently used position) without evicting
3. If it doesn't exist and the cache is full, evicts the least recently used item
4. Then sets/updates the value