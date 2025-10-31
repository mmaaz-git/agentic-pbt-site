# Bug Report: LRU Cache Incorrectly Evicts on Update

**Target**: `dask.dataframe.dask_expr._util.LRU`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The LRU cache incorrectly evicts an item when updating an existing key in a full cache, causing the cache size to drop below maxsize.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings

@given(st.integers(min_value=1, max_value=100), st.text())
@settings(max_examples=500)
def test_lru_update_preserves_size(maxsize, key):
    from dask.dataframe.dask_expr._util import LRU

    lru = LRU(maxsize)

    for i in range(maxsize):
        lru[f'key_{i}'] = i

    assert len(lru) == maxsize

    lru[f'key_0'] = 999

    assert len(lru) == maxsize, f"Updating existing key should preserve size {maxsize}, got {len(lru)}"
```

**Failing input**: Any maxsize >= 1 and any key that exists in a full LRU.

## Reproducing the Bug

```python
from dask.dataframe.dask_expr._util import LRU

lru = LRU(3)
lru['a'] = 1
lru['b'] = 2
lru['c'] = 3

print(f"Initial: {list(lru.keys())}, len={len(lru)}")

lru['c'] = 4

print(f"After update: {list(lru.keys())}, len={len(lru)}")
```

Output:
```
Initial: ['a', 'b', 'c'], len=3
After update: ['b', 'c'], len=2
```

Expected: The cache should still contain 3 items after updating 'c'.

## Why This Is A Bug

The `__setitem__` method checks if `len(self) >= self.maxsize` before any operation. When updating an existing key in a full cache:

1. The condition `len(self) >= maxsize` is True
2. An item is evicted (reducing size to maxsize-1)
3. The existing key is updated (size remains maxsize-1)

This violates the cache's invariant that it should contain up to maxsize items. The cache ends up with fewer items than allowed.

## Fix

The fix is to check whether the key already exists before deciding to evict:

```diff
diff --git a/dask/dataframe/dask_expr/_util.py b/dask/dataframe/dask_expr/_util.py
index ...
--- a/dask/dataframe/dask_expr/_util.py
+++ b/dask/dataframe/dask_expr/_util.py
@@ -108,7 +108,7 @@ class LRU(UserDict[K, V]):
         return value

     def __setitem__(self, key: K, value: V) -> None:
-        if len(self) >= self.maxsize:
+        if key not in self.data and len(self) >= self.maxsize:
             cast(OrderedDict, self.data).popitem(last=False)
         super().__setitem__(key, value)
```

This ensures eviction only happens when adding a new key would exceed maxsize, not when updating an existing key.