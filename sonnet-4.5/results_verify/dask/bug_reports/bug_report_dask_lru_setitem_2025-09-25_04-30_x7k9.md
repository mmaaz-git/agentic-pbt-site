# Bug Report: LRU Cache Incorrectly Evicts on Update

**Target**: `dask.dataframe.dask_expr._util.LRU.__setitem__`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The LRU cache incorrectly evicts items when updating an existing key. The `__setitem__` method checks if the cache is full before checking if the key already exists, causing it to unnecessarily evict the least recently used item even when just updating a value.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from dask.dataframe.dask_expr._util import LRU

@given(st.integers(min_value=1, max_value=10))
@settings(max_examples=200)
def test_lru_getitem_moves_to_end(maxsize):
    lru = LRU(maxsize)
    for i in range(maxsize):
        lru[i] = i * 2

    lru[0]

    lru[maxsize] = maxsize * 2

    assert 0 in lru
```

**Failing input**: `maxsize=1`

## Reproducing the Bug

```python
from dask.dataframe.dask_expr._util import LRU

lru = LRU(2)
lru["a"] = 1
lru["b"] = 2

lru["a"] = 999

assert len(lru) == 2
assert lru["a"] == 999
assert lru["b"] == 2
```

This fails because updating `"a"` triggers eviction of `"b"`, leaving only one item in the cache instead of two.

## Why This Is A Bug

According to the docstring, LRU should evict "the least recently looked-up key when full". When updating an existing key, the cache is not actually full - we're just replacing a value, not adding a new entry. The current implementation incorrectly evicts an item every time `__setitem__` is called when at max capacity, regardless of whether it's an update or a new insertion.

This violates the fundamental LRU cache invariant that `len(cache) <= maxsize` should be maintained while maximizing the number of stored items.

## Fix

```diff
--- a/dask/dataframe/dask_expr/_util.py
+++ b/dask/dataframe/dask_expr/_util.py
@@ -107,7 +107,7 @@ class LRU(UserDict[K, V]):
         return value

     def __setitem__(self, key: K, value: V) -> None:
-        if len(self) >= self.maxsize:
+        if key not in self and len(self) >= self.maxsize:
             cast(OrderedDict, self.data).popitem(last=False)
         super().__setitem__(key, value)
```