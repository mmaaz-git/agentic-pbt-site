# Bug Report: dask.dataframe.dask_expr._util.LRU Incorrect Eviction on Update

**Target**: `dask.dataframe.dask_expr._util.LRU`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The LRU cache implementation incorrectly evicts items when updating an existing key, violating the fundamental LRU invariant that eviction should only occur when adding new keys that exceed the cache's maximum size.

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

    assert set(lru.keys()) == set(access_order), f"LRU keys {set(lru.keys())} != expected {set(access_order)}"


if __name__ == "__main__":
    test_lru_eviction_order()
```

<details>

<summary>
**Failing input**: `maxsize=2, operations=[('set', 1), ('set', 0), ('set', 0)]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 38, in <module>
    test_lru_eviction_order()
    ~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 6, in test_lru_eviction_order
    maxsize=st.integers(min_value=1, max_value=10),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 34, in test_lru_eviction_order
    assert set(lru.keys()) == set(access_order), f"LRU keys {set(lru.keys())} != expected {set(access_order)}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: LRU keys {0} != expected {0, 1}
Falsifying example: test_lru_eviction_order(
    maxsize=2,
    operations=[('set', 1), ('set', 0), ('set', 0)],
)
```
</details>

## Reproducing the Bug

```python
from dask.dataframe.dask_expr._util import LRU

lru = LRU(maxsize=2)
lru[0] = 0
lru[1] = 1
print(f"After adding 0 and 1: len={len(lru)}, keys={list(lru.keys())}")
assert len(lru) == 2, f"Expected len=2, got {len(lru)}"
assert 0 in lru, "Key 0 should be in cache"
assert 1 in lru, "Key 1 should be in cache"

# Update existing key 1
lru[1] = 1
print(f"After updating key 1: len={len(lru)}, keys={list(lru.keys())}")
assert len(lru) == 2, f"Expected len=2, got {len(lru)}"
assert 0 in lru, "Key 0 should still be in cache after updating key 1"
assert 1 in lru, "Key 1 should still be in cache"
```

<details>

<summary>
AssertionError: Cache size reduced from 2 to 1 after updating existing key
</summary>
```
After adding 0 and 1: len=2, keys=[0, 1]
After updating key 1: len=1, keys=[1]
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/28/repo.py", line 14, in <module>
    assert len(lru) == 2, f"Expected len=2, got {len(lru)}"
           ^^^^^^^^^^^^^
AssertionError: Expected len=2, got 1
```
</details>

## Why This Is A Bug

This implementation violates the fundamental LRU cache contract documented in the class docstring: "Limited size mapping, evicting the least recently looked-up key when full". The cache should only evict items when it becomes full due to adding a NEW key that would exceed maxsize.

The bug occurs because the `__setitem__` method (lines 109-112 in `/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/dataframe/dask_expr/_util.py`) checks `if len(self) >= self.maxsize` and evicts an item BEFORE checking whether the key already exists in the cache. When updating an existing key with a cache at maxsize, this causes an unnecessary eviction, reducing the cache size below its maximum capacity.

This breaks three critical expectations:
1. **Cache capacity invariant**: A cache with maxsize=N should maintain N items when possible
2. **Update semantics**: Updating an existing key should not change the cache size
3. **LRU ordering**: Updates should move items to the end without eviction

The LRU cache is used internally by `_BackendData` (line 124) to cache division information with maxsize=10. Incorrect evictions can lead to unnecessary cache misses and performance degradation in dask-expr operations that rely on cached division data.

## Relevant Context

The LRU class inherits from `UserDict` and uses an `OrderedDict` internally for maintaining insertion order. The `__getitem__` method correctly implements LRU behavior by moving accessed items to the end via `move_to_end(key)`. However, `__setitem__` fails to distinguish between inserting a new key (which may require eviction) and updating an existing key (which should never trigger eviction).

This is a standard LRU cache implementation pattern issue where the eviction logic must be conditional on whether the operation adds a new key. The Python standard library's `functools.lru_cache` and other robust implementations handle this correctly.

Code location: `/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/dataframe/dask_expr/_util.py:96-113`

## Proposed Fix

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