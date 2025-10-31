# Bug Report: dask.dataframe.dask_expr LRU Cache Incorrect Eviction

**Target**: `dask.dataframe.dask_expr._util.LRU`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The LRU cache incorrectly evicts items when updating an existing key, even when the cache is not full. This violates the LRU invariant that the cache should only evict items when it exceeds maxsize with a NEW insertion.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
from dask.dataframe.dask_expr._util import LRU


@given(
    maxsize=st.integers(min_value=1, max_value=10),
    operations=st.lists(
        st.tuples(
            st.sampled_from(['set', 'get']),
            st.integers(min_value=0, max_value=20)
        ),
        min_size=0,
        max_size=100
    )
)
@settings(max_examples=500)
def test_lru_eviction_order(maxsize, operations):
    lru = LRU(maxsize)
    access_order = []

    for op_type, key in operations:
        if op_type == 'set':
            lru[key] = key
            if key in access_order:
                access_order.remove(key)
            access_order.append(key)
            if len(access_order) > maxsize:
                access_order.pop(0)
        elif op_type == 'get' and key in lru:
            _ = lru[key]
            access_order.remove(key)
            access_order.append(key)

    assert set(lru.keys()) == set(access_order)
```

**Failing input**: `maxsize=2, operations=[('set', 0), ('set', 1), ('set', 1)]`

## Reproducing the Bug

```python
from dask.dataframe.dask_expr._util import LRU

lru = LRU(maxsize=2)
lru[0] = 0
lru[1] = 1
assert len(lru) == 2
assert 0 in lru
assert 1 in lru

lru[1] = 1
assert len(lru) == 2
assert 0 in lru
```

The final assertion fails because key 0 was incorrectly evicted when updating key 1, leaving only `{1: 1}` in the cache.

## Why This Is A Bug

An LRU cache should maintain the invariant that it contains at most `maxsize` items. When updating an existing key, the cache size does not increase, so no eviction should occur. The current implementation checks `if len(self) >= self.maxsize` before checking if the key already exists, causing unnecessary evictions.

This violates the documented behavior: "Limited size mapping, evicting the least recently looked-up key when full". The cache is not full when we're updating an existing key.

## Fix

```diff
--- a/dask/dataframe/dask_expr/_util.py
+++ b/dask/dataframe/dask_expr/_util.py
@@ -107,7 +107,9 @@ class LRU(UserDict[K, V]):
         return value

     def __setitem__(self, key: K, value: V) -> None:
-        if len(self) >= self.maxsize:
+        if key in self:
+            cast(OrderedDict, self.data).move_to_end(key)
+        elif len(self) >= self.maxsize:
             cast(OrderedDict, self.data).popitem(last=False)
         super().__setitem__(key, value)
```

This fix:
1. Checks if the key already exists before evicting
2. If it exists, moves it to the end (LRU semantics for updates)
3. Only evicts when adding a NEW key that would exceed maxsize