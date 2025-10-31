# Bug Report: django.core.cache.backends.locmem MAX_ENTRIES Violation

**Target**: `django.core.cache.backends.locmem.LocMemCache`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The LocMemCache backend can exceed its configured MAX_ENTRIES limit when MAX_ENTRIES is small relative to CULL_FREQUENCY. This violates the documented constraint that the cache should not hold more than MAX_ENTRIES items.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings as hyp_settings
from django.core.cache.backends.locmem import LocMemCache


@hyp_settings(max_examples=200)
@given(
    st.integers(min_value=1, max_value=10),
    st.integers(min_value=2, max_value=10)
)
def test_max_entries_never_exceeded(max_entries, cull_frequency):
    cache = LocMemCache(f'test_{max_entries}_{cull_frequency}', {
        'OPTIONS': {'MAX_ENTRIES': max_entries, 'CULL_FREQUENCY': cull_frequency}
    })
    cache.clear()

    num_to_add = max_entries * 2
    for i in range(num_to_add):
        cache.set(f'key_{i}', i)
        cache_size = len(cache._cache)
        assert cache_size <= max_entries
```

**Failing input**: `max_entries=2, cull_frequency=3` (and any case where `max_entries < cull_frequency`)

## Reproducing the Bug

```python
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'test_settings')

import django
from django.conf import settings
if not settings.configured:
    settings.configure(DEBUG=True, DATABASES={})
    django.setup()

from django.core.cache.backends.locmem import LocMemCache

cache = LocMemCache('test', {
    'OPTIONS': {'MAX_ENTRIES': 2, 'CULL_FREQUENCY': 3}
})
cache.clear()

for i in range(3):
    cache.set(f'key_{i}', i)
    print(f"After adding key_{i}: cache size = {len(cache._cache)}")

print(f"\nFinal: cache has {len(cache._cache)} entries, MAX_ENTRIES is {cache._max_entries}")
```

Output:
```
After adding key_0: cache size = 1
After adding key_1: cache size = 2
After adding key_2: cache size = 3

Final: cache has 3 entries, MAX_ENTRIES is 2
```

## Why This Is A Bug

The cache violates its own documented constraint. From `base.py` lines 71-75, MAX_ENTRIES is explicitly defined as the maximum number of entries allowed in the cache. The LocMemCache implementation should enforce this limit but fails to do so in certain configurations.

The root cause is in `locmem.py` line 97:

```python
count = len(self._cache) // self._cull_frequency
```

When `len(self._cache) < self._cull_frequency`, integer division returns 0, meaning no items are culled. The subsequent code then adds a new item, exceeding MAX_ENTRIES.

## Fix

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

This ensures at least one item is always culled when the cache is full, preventing it from exceeding MAX_ENTRIES.