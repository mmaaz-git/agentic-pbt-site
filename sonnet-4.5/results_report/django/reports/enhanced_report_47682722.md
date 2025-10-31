# Bug Report: django.core.cache.backends.locmem.LocMemCache Max Entries Limit Violation

**Target**: `django.core.cache.backends.locmem.LocMemCache._cull`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The Django LocMemCache can exceed its configured `max_entries` limit when the culling calculation `len(cache) // cull_frequency` results in zero, allowing unbounded cache growth beyond the specified maximum.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from django.core.cache.backends.locmem import LocMemCache
import itertools

counter = itertools.count()


@settings(max_examples=200)
@given(
    st.integers(min_value=1, max_value=100),
    st.integers(min_value=1, max_value=10)
)
def test_max_entries_enforcement(max_entries, extra):
    cache = LocMemCache(f"test_{next(counter)}", {
        "timeout": 300,
        "max_entries": max_entries,
        "cull_frequency": 3
    })

    total_items = max_entries + extra
    for i in range(total_items):
        cache.set(f"key_{i}", i)

    current_size = len(cache._cache)
    assert current_size <= max_entries, \
        f"Cache size {current_size} exceeds max_entries {max_entries}"
```

<details>

<summary>
**Failing input**: `max_entries=1, extra=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 30, in <module>
    test_max_entries_enforcement()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 9, in test_max_entries_enforcement
    @given(

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 25, in test_max_entries_enforcement
    assert current_size <= max_entries, \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Cache size 2 exceeds max_entries 1
Falsifying example: test_max_entries_enforcement(
    max_entries=1,
    extra=1,
)
```
</details>

## Reproducing the Bug

```python
from django.core.cache.backends.locmem import LocMemCache

# Create a cache with max_entries=1 and cull_frequency=3
cache = LocMemCache("test", {
    "timeout": 300,
    "max_entries": 1,
    "cull_frequency": 3
})

print("Initial state:")
print(f"Cache size: {len(cache._cache)}")
print(f"Max entries: {cache._max_entries}")
print(f"Cull frequency: {cache._cull_frequency}")
print()

# Add the first item
cache.set("key1", "value1")
print("After adding key1:")
print(f"Cache size: {len(cache._cache)}")
print(f"Cache contents: {list(cache._cache.keys())}")
print()

# Add the second item - this should trigger culling
cache.set("key2", "value2")
print("After adding key2:")
print(f"Cache size: {len(cache._cache)}")
print(f"Max entries: {cache._max_entries}")
print(f"Cache contents: {list(cache._cache.keys())}")
print()

# Demonstrate the culling calculation
print("Culling calculation when cache size was 1:")
print(f"len(cache) // cull_frequency = {1} // {cache._cull_frequency} = {1 // cache._cull_frequency}")
print(f"Number of items to remove: {1 // cache._cull_frequency}")
print()

if len(cache._cache) > cache._max_entries:
    print(f"ERROR: Cache size ({len(cache._cache)}) exceeds max_entries ({cache._max_entries})")
else:
    print("Cache size is within limits")
```

<details>

<summary>
ERROR: Cache size (2) exceeds max_entries (1)
</summary>
```
Initial state:
Cache size: 0
Max entries: 1
Cull frequency: 3

After adding key1:
Cache size: 1
Cache contents: [':1:key1']

After adding key2:
Cache size: 2
Max entries: 1
Cache contents: [':1:key2', ':1:key1']

Culling calculation when cache size was 1:
len(cache) // cull_frequency = 1 // 3 = 0
Number of items to remove: 0

ERROR: Cache size (2) exceeds max_entries (1)
```
</details>

## Why This Is A Bug

The parameter name `max_entries` creates a clear contractual expectation that the cache will not exceed this number of entries. This expectation is violated due to a mathematical oversight in the culling implementation.

The bug occurs in the interaction between `_set()` (locmem.py:45-50) and `_cull()` (locmem.py:92-100):

1. When adding a new entry, `_set()` checks if `len(self._cache) >= self._max_entries`
2. If true, it calls `_cull()` to make room
3. `_cull()` calculates items to remove as: `count = len(self._cache) // self._cull_frequency`
4. When cache size < cull_frequency, integer division yields 0 (e.g., 1 // 3 = 0)
5. Zero items are removed, but the new item is still added
6. Result: Cache size exceeds max_entries

This violates the fundamental purpose of the `max_entries` parameter, which is to limit memory consumption. While Django documentation notes that culling removes "1/3 of entries" when CULL_FREQUENCY=3, it's reasonable to expect that `max_entries` represents a hard limit that should never be exceeded.

## Relevant Context

The bug manifests when `max_entries < cull_frequency`. While the default configuration (max_entries=300, cull_frequency=3) doesn't trigger this issue, any configuration where the cache limit is smaller than the culling divisor will exhibit this behavior.

Key code locations:
- Cache initialization: `/django/core/cache/backends/base.py:71-81` (sets default max_entries=300, cull_frequency=3)
- Entry addition logic: `/django/core/cache/backends/locmem.py:45-50` (_set method)
- Culling implementation: `/django/core/cache/backends/locmem.py:92-100` (_cull method)

Django documentation reference: https://docs.djangoproject.com/en/stable/topics/cache/#local-memory-caching

The documentation states that LocMemCache uses an "LRU culling strategy" and that when MAX_ENTRIES is reached, "it will cull a fraction of entries," but doesn't explicitly guarantee the cache will never exceed max_entries. However, the parameter name itself strongly implies this constraint.

## Proposed Fix

```diff
--- a/django/core/cache/backends/locmem.py
+++ b/django/core/cache/backends/locmem.py
@@ -94,7 +94,7 @@ class LocMemCache(BaseCache):
             self._cache.clear()
             self._expire_info.clear()
         else:
-            count = len(self._cache) // self._cull_frequency
+            count = max(1, len(self._cache) // self._cull_frequency)
             for i in range(count):
                 key, _ = self._cache.popitem()
                 del self._expire_info[key]
```