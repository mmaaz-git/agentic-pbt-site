# Bug Report: dask.dataframe.dask_expr._util.LRU Incorrectly Evicts Items on Update

**Target**: `dask.dataframe.dask_expr._util.LRU`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The LRU cache implementation incorrectly evicts an item when updating an existing key in a full cache, causing the cache size to drop below its maximum capacity and violating standard LRU cache semantics.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings

@given(st.integers(min_value=2, max_value=100))
@settings(max_examples=500, deadline=None)
def test_lru_update_preserves_size(maxsize):
    from dask.dataframe.dask_expr._util import LRU

    lru = LRU(maxsize)

    # Fill the cache to capacity
    for i in range(maxsize):
        lru[f'key_{i}'] = i

    assert len(lru) == maxsize

    # Update an existing key (not the first one)
    # We pick a middle key to ensure it's not the oldest
    update_key = f'key_{maxsize // 2}'
    lru[update_key] = 999

    assert len(lru) == maxsize, f"Updating existing key should preserve size {maxsize}, got {len(lru)}"

if __name__ == "__main__":
    # Run the test
    test_lru_update_preserves_size()
```

<details>

<summary>
**Failing input**: `maxsize=2`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/57/hypo.py", line 25, in <module>
    test_lru_update_preserves_size()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/57/hypo.py", line 4, in test_lru_update_preserves_size
    @settings(max_examples=500, deadline=None)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/57/hypo.py", line 21, in test_lru_update_preserves_size
    assert len(lru) == maxsize, f"Updating existing key should preserve size {maxsize}, got {len(lru)}"
           ^^^^^^^^^^^^^^^^^^^
AssertionError: Updating existing key should preserve size 2, got 1
Falsifying example: test_lru_update_preserves_size(
    maxsize=2,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
from dask.dataframe.dask_expr._util import LRU

# Create an LRU cache with maxsize of 3
lru = LRU(3)

# Fill the cache to capacity
lru['a'] = 1
lru['b'] = 2
lru['c'] = 3

print(f"Initial state after filling cache:")
print(f"  Keys: {list(lru.keys())}")
print(f"  Length: {len(lru)}")
print(f"  Expected length: 3")
print()

# Update an existing key
print(f"Updating existing key 'c' with new value 4...")
lru['c'] = 4

print(f"State after updating 'c':")
print(f"  Keys: {list(lru.keys())}")
print(f"  Length: {len(lru)}")
print(f"  Expected length: 3")
print()

# Check what happened
if len(lru) < 3:
    print(f"ERROR: Cache size dropped from 3 to {len(lru)} after updating existing key!")
    print(f"Missing key(s): {set(['a', 'b', 'c']) - set(lru.keys())}")
else:
    print(f"Cache size correctly maintained at {len(lru)}")
```

<details>

<summary>
Output demonstrating the bug
</summary>
```
Initial state after filling cache:
  Keys: ['a', 'b', 'c']
  Length: 3
  Expected length: 3

Updating existing key 'c' with new value 4...
State after updating 'c':
  Keys: ['b', 'c']
  Length: 2
  Expected length: 3

ERROR: Cache size dropped from 3 to 2 after updating existing key!
Missing key(s): {'a'}
```
</details>

## Why This Is A Bug

The bug violates fundamental LRU cache semantics. When updating an existing key in a full LRU cache, the cache should maintain its size at maxsize, not reduce it. The current implementation in `/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/dataframe/dask_expr/_util.py` has a logical error in the `__setitem__` method (lines 109-112):

```python
def __setitem__(self, key: K, value: V) -> None:
    if len(self) >= self.maxsize:
        cast(OrderedDict, self.data).popitem(last=False)
    super().__setitem__(key, value)
```

The bug occurs because:
1. When the cache is full (`len(self) >= self.maxsize`), the condition evaluates to True
2. The oldest item is evicted via `popitem(last=False)`, reducing the cache size by 1
3. The existing key is then updated (which doesn't increase the size since it already exists)
4. Result: The cache now has `maxsize - 1` items instead of `maxsize`

This behavior contradicts standard LRU cache behavior where:
- Updating an existing key should only move it to the most recently used position
- The cache size should remain constant when updating existing keys
- Eviction should only occur when adding NEW keys that would exceed maxsize

## Relevant Context

The LRU class is used internally in Dask for caching division information (see line 124 in the same file where `_BackendData` initializes `self._division_info = LRU(10)`). This incorrect behavior reduces cache effectiveness by maintaining fewer items than the specified capacity, potentially leading to more cache misses and reduced performance in Dask dataframe operations.

The bug affects any code path that updates existing cache entries when the cache is at capacity. While users don't directly interact with this internal cache, the performance implications affect all Dask dataframe operations that rely on cached division information.

## Proposed Fix

The fix is to check whether the key already exists before deciding to evict:

```diff
--- a/dask/dataframe/dask_expr/_util.py
+++ b/dask/dataframe/dask_expr/_util.py
@@ -109,7 +109,7 @@ class LRU(UserDict[K, V]):
         return value

     def __setitem__(self, key: K, value: V) -> None:
-        if len(self) >= self.maxsize:
+        if key not in self.data and len(self) >= self.maxsize:
             cast(OrderedDict, self.data).popitem(last=False)
         super().__setitem__(key, value)
```