# Bug Report: django.core.cache.backends.locmem - MAX_ENTRIES Violation

**Target**: `django.core.cache.backends.locmem.LocMemCache`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The LocMemCache backend violates its MAX_ENTRIES constraint when `cull_frequency > max_entries`. This causes the cache to grow beyond the configured maximum size, potentially leading to unbounded memory growth.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
from django.core.cache.backends.locmem import LocMemCache


def make_cache(max_entries=300, cull_frequency=3):
    return LocMemCache('test_edge', {
        'TIMEOUT': 300,
        'OPTIONS': {
            'MAX_ENTRIES': max_entries,
            'CULL_FREQUENCY': cull_frequency,
        },
        'KEY_PREFIX': 'test',
        'VERSION': 1,
    })


@given(st.integers(min_value=1, max_value=30), st.integers(min_value=1, max_value=10))
@settings(max_examples=300)
def test_exact_max_entries_boundary(max_entries, num_additional):
    cache = make_cache(max_entries=max_entries, cull_frequency=3)
    cache.clear()

    for i in range(max_entries + num_additional):
        cache.set(f"key_{i:04d}", i)

    final_size = len(cache._cache)
    assert final_size <= max_entries, \
        f"Cache size {final_size} exceeds max_entries {max_entries}"
```

**Failing input**: `max_entries=1, num_additional=1` (or any value where cull_frequency >= max_entries)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.core.cache.backends.locmem import LocMemCache

cache = LocMemCache('reproduce_bug', {
    'TIMEOUT': 300,
    'OPTIONS': {
        'MAX_ENTRIES': 1,
        'CULL_FREQUENCY': 3,
    },
    'KEY_PREFIX': 'test',
    'VERSION': 1,
})
cache.clear()

cache.set("key1", "value1")
print(f"After setting key1: cache size = {len(cache._cache)}")

cache.set("key2", "value2")
print(f"After setting key2: cache size = {len(cache._cache)}")

assert len(cache._cache) <= cache._max_entries, \
    f"BUG: Cache has {len(cache._cache)} entries, exceeding max_entries={cache._max_entries}"
```

**Output:**
```
After setting key1: cache size = 1
After setting key2: cache size = 2
AssertionError: BUG: Cache has 2 entries, exceeding max_entries=1
```

## Why This Is A Bug

The `_cull` method in `locmem.py` (lines 92-100) calculates how many items to remove using integer division:

```python
def _cull(self):
    if self._cull_frequency == 0:
        self._cache.clear()
        self._expire_info.clear()
    else:
        count = len(self._cache) // self._cull_frequency  # BUG IS HERE
        for i in range(count):
            key, _ = self._cache.popitem()
            del self._expire_info[key]
```

When `cull_frequency > len(cache)`, the integer division `len(self._cache) // self._cull_frequency` evaluates to 0, meaning no items are removed. However, `_set` (lines 45-50) then adds the new item, causing the cache to exceed `max_entries`.

Example with `max_entries=1, cull_frequency=3`:
1. Cache has 1 entry (at capacity)
2. Try to add second entry
3. Check: `len(cache) >= max_entries` → `1 >= 1` → True, so cull
4. Cull: `count = 1 // 3 = 0` items removed
5. Add new entry: cache now has 2 entries
6. **Violates invariant: `len(cache) <= max_entries`**

This violates the documented behavior that MAX_ENTRIES controls the maximum number of entries allowed in the cache.

## Fix

The cull logic should ensure at least one item is removed when the cache is at capacity, or enforce a minimum cull count that maintains the max_entries constraint:

```diff
--- a/django/core/cache/backends/locmem.py
+++ b/django/core/cache/backends/locmem.py
@@ -94,7 +94,8 @@ class LocMemCache(BaseCache):
             self._cache.clear()
             self._expire_info.clear()
         else:
-            count = len(self._cache) // self._cull_frequency
+            # Ensure we remove at least 1 item to maintain max_entries constraint
+            count = max(1, len(self._cache) // self._cull_frequency)
             for i in range(count):
                 key, _ = self._cache.popitem()
                 del self._expire_info[key]
```

This ensures that when the cache is at capacity and culling is triggered, at least one item is removed before adding the new item, maintaining the `max_entries` invariant.