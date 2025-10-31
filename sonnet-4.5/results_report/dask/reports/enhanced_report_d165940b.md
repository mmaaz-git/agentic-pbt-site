# Bug Report: dask.dataframe.dask_expr._util.LRU Unnecessarily Evicts Items When Updating Existing Keys

**Target**: `dask.dataframe.dask_expr._util.LRU.__setitem__`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The LRU cache implementation incorrectly evicts the least recently used item when updating an existing key in a full cache, reducing the cache size unnecessarily and degrading performance.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from dask.dataframe.dask_expr._util import LRU

@given(st.integers(min_value=2, max_value=10))
@settings(max_examples=200)
def test_lru_update_evicts_when_it_shouldnt(maxsize):
    """Test that updating an existing key evicts the least recently used item unnecessarily."""
    lru = LRU(maxsize)

    # Fill the cache
    for i in range(maxsize):
        lru[i] = i * 10

    # Track which key is least recently used (should be 0)
    least_recent_key = 0

    # Update the LAST key (which is NOT the least recently used)
    last_key = maxsize - 1
    lru[last_key] = 999

    # BUG: The least recently used key (0) gets evicted even though we're just updating
    # an existing key, not adding a new one
    assert least_recent_key in lru, f"Key {least_recent_key} was evicted when updating existing key {last_key} (maxsize={maxsize})"

if __name__ == "__main__":
    test_lru_update_evicts_when_it_shouldnt()
```

<details>

<summary>
**Failing input**: `maxsize=2`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/45/hypo.py", line 26, in <module>
    test_lru_update_evicts_when_it_shouldnt()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/45/hypo.py", line 5, in test_lru_update_evicts_when_it_shouldnt
    @settings(max_examples=200)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/45/hypo.py", line 23, in test_lru_update_evicts_when_it_shouldnt
    assert least_recent_key in lru, f"Key {least_recent_key} was evicted when updating existing key {last_key} (maxsize={maxsize})"
           ^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Key 0 was evicted when updating existing key 1 (maxsize=2)
Falsifying example: test_lru_update_evicts_when_it_shouldnt(
    maxsize=2,
)
```
</details>

## Reproducing the Bug

```python
from dask.dataframe.dask_expr._util import LRU

# Test 1: Basic update behavior
print("Test 1: Basic update behavior")
lru = LRU(2)
lru["a"] = 1
lru["b"] = 2
print(f"Initial state: {dict(lru)}")
print(f"Cache size: {len(lru)}")

# Update existing key "a"
lru["a"] = 999
print(f"After updating 'a': {dict(lru)}")
print(f"Cache size: {len(lru)}")

# What we expect: Both "a" and "b" should still be in the cache
# What actually happens: We'll see...
print(f"'a' in cache: {('a' in lru)}")
print(f"'b' in cache: {('b' in lru)}")
if 'a' in lru:
    print(f"Value of 'a': {lru['a']}")
if 'b' in lru:
    print(f"Value of 'b': {lru['b']}")

print("\n" + "="*50 + "\n")

# Test 2: Update in a size-1 cache
print("Test 2: Update in a size-1 cache")
lru = LRU(1)
lru["x"] = 100
print(f"Initial: {dict(lru)}")
lru["x"] = 200  # Update same key
print(f"After update: {dict(lru)}")
print(f"'x' in cache: {('x' in lru)}")
if 'x' in lru:
    print(f"Value of 'x': {lru['x']}")

print("\n" + "="*50 + "\n")

# Test 3: Demonstrating the eviction issue
print("Test 3: Eviction during update")
lru = LRU(3)
lru["first"] = 1
lru["second"] = 2
lru["third"] = 3
print(f"Full cache: {dict(lru)}")

# Access "first" to make it recently used
_ = lru["first"]
print(f"After accessing 'first': {list(lru.keys())}")

# Update "third" - should NOT evict anything since we're not adding a new key
lru["third"] = 300
print(f"After updating 'third': {dict(lru)}")
print(f"Cache size: {len(lru)}")

# Check what got evicted (if anything)
for key in ["first", "second", "third"]:
    print(f"'{key}' in cache: {(key in lru)}")
```

<details>

<summary>
Test output showing unnecessary eviction during update
</summary>
```
Test 1: Basic update behavior
Initial state: {'a': 1, 'b': 2}
Cache size: 2
After updating 'a': {'b': 2, 'a': 999}
Cache size: 2
'a' in cache: True
'b' in cache: True
Value of 'a': 999
Value of 'b': 2

==================================================

Test 2: Update in a size-1 cache
Initial: {'x': 100}
After update: {'x': 200}
'x' in cache: True
Value of 'x': 200

==================================================

Test 3: Eviction during update
Full cache: {'first': 1, 'second': 2, 'third': 3}
After accessing 'first': ['second', 'third', 'first']
After updating 'third': {'third': 300, 'first': 1}
Cache size: 2
'first' in cache: True
'second' in cache: False
'third' in cache: True
```
</details>

## Why This Is A Bug

This violates standard LRU cache behavior in multiple ways:

1. **Incorrect Eviction Logic**: The `__setitem__` method checks `if len(self) >= self.maxsize` and evicts BEFORE checking if the key already exists. This means updating any existing key in a full cache will always trigger an unnecessary eviction.

2. **Violates LRU Semantics**: According to the class docstring, the cache should evict "the least recently looked-up key when full". However, when updating an existing key, the cache isn't becoming fuller - we're just changing a value. Standard LRU implementations (like Python's `functools.lru_cache`) only evict when adding NEW keys that would exceed capacity.

3. **Performance Degradation**: Every update to a full cache causes:
   - Unnecessary eviction of the least recently used item
   - Re-insertion of the updated key
   - Potential cache miss for the evicted item on next access

4. **Size Inconsistency**: In Test 3, updating "third" in a full 3-item cache reduces it to 2 items, even though we're just changing an existing value. The cache should maintain its maximum size when possible.

## Relevant Context

The bug is in `/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/dataframe/dask_expr/_util.py` at lines 109-112.

The LRU class inherits from `UserDict` and uses an `OrderedDict` internally. The `__getitem__` method correctly moves accessed items to the end (making them most recently used), but `__setitem__` doesn't distinguish between updates and insertions.

This is an internal utility class used by Dask's dataframe expressions module, particularly in the `_BackendData` class for caching division information. While not part of the public API, the inefficiency could impact performance in data processing pipelines that frequently update cached values.

Documentation: The class has minimal documentation, with only a one-line docstring. There's no explicit specification of update vs. insert behavior, but standard LRU cache conventions apply.

## Proposed Fix

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