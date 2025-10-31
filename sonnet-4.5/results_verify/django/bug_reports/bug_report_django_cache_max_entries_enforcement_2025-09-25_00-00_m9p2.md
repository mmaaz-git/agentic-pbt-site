# Bug Report: django.core.cache.backends.locmem.LocMemCache Max Entries Enforcement

**Target**: `django.core.cache.backends.locmem.LocMemCache._set`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The local memory cache can exceed its configured `max_entries` limit when `cull_frequency` is set such that the number of items to remove rounds down to zero (i.e., when `len(cache) // cull_frequency == 0`).

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

**Failing input**: `max_entries=1, extra=1`

## Reproducing the Bug

```python
from django.core.cache.backends.locmem import LocMemCache

cache = LocMemCache("test", {
    "timeout": 300,
    "max_entries": 1,
    "cull_frequency": 3
})

cache.set("key1", "value1")
cache.set("key2", "value2")

print(f"Cache size: {len(cache._cache)}")
print(f"Max entries: {cache._max_entries}")
print(f"Cache contents: {list(cache._cache.keys())}")
```

Expected: Cache size should be <= 1
Actual: Cache size is 2 (exceeds max_entries)

## Why This Is A Bug

The cache is configured with `max_entries` to limit memory usage. When this limit is exceeded, the cache should not continue growing unbounded. The current implementation fails to enforce this constraint when the cull calculation `len(cache) // cull_frequency` equals zero.

The bug occurs in the `_set` method:

```python
def _set(self, key, value, timeout=DEFAULT_TIMEOUT):
    if len(self._cache) >= self._max_entries:
        self._cull()
    self._cache[key] = value
    ...
```

And in `_cull`:

```python
def _cull(self):
    if self._cull_frequency == 0:
        self._cache.clear()
        self._expire_info.clear()
    else:
        count = len(self._cache) // self._cull_frequency  # Integer division rounds down!
        for i in range(count):
            key, _ = self._cache.popitem()
            del self._expire_info[key]
```

When `len(self._cache) < self._cull_frequency`, the count becomes 0, and no items are removed. The new item is then added, exceeding `max_entries`.

## Fix

```diff
--- a/django/core/cache/backends/locmem.py
+++ b/django/core/cache/backends/locmem.py
@@ -78,7 +78,7 @@ class LocMemCache(BaseCache):
         if self._cull_frequency == 0:
             self._cache.clear()
             self._expire_info.clear()
         else:
-            count = len(self._cache) // self._cull_frequency
+            count = max(1, len(self._cache) // self._cull_frequency)
             for i in range(count):
                 key, _ = self._cache.popitem()
                 del self._expire_info[key]
```

This ensures that at least one item is removed when culling is triggered, preventing the cache from exceeding `max_entries`.