# Bug Report: django.core.cache LocMemCache._delete() Incomplete Cleanup

**Target**: `django.core.cache.backends.locmem.LocMemCache._delete()`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_delete()` method in `LocMemCache` fails to clean up `_expire_info` when a key exists in `_expire_info` but not in `_cache`, leading to an inconsistent internal state and potential memory leak.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.core.cache.backends.locmem import LocMemCache
import time

@given(st.text(min_size=1, max_size=100))
def test_delete_maintains_cache_expire_info_consistency(key):
    cache = LocMemCache('test', {})
    cache.clear()

    cache_key = cache.make_key(key, version=1)
    cache._expire_info[cache_key] = time.time() + 100

    cache._delete(cache_key)

    # Property: After _delete(), key should not be in either dictionary
    assert cache_key not in cache._cache
    assert cache_key not in cache._expire_info  # FAILS
```

**Failing input**: Any valid cache key

## Reproducing the Bug

```python
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'test_settings')

import django
from django.conf import settings
if not settings.configured:
    settings.configure(
        DEBUG=True,
        DATABASES={},
        CACHES={
            'default': {
                'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
                'LOCATION': 'test',
            }
        }
    )
    django.setup()

from django.core.cache.backends.locmem import LocMemCache
import time

cache = LocMemCache('test', {})
cache_key = ':1:testkey'

cache._expire_info[cache_key] = time.time() + 100

print(f"Before: key in _expire_info = {cache_key in cache._expire_info}")

cache._delete(cache_key)

print(f"After: key in _expire_info = {cache_key in cache._expire_info}")
```

**Expected output**: `After: key in _expire_info = False`

**Actual output**: `After: key in _expire_info = True`

## Why This Is A Bug

The `_delete()` method is supposed to maintain the invariant that `_cache` and `_expire_info` are always synchronized. However, the current implementation fails to remove keys from `_expire_info` when the key is not present in `_cache`:

```python
def _delete(self, key):
    try:
        del self._cache[key]       # If this raises KeyError...
        del self._expire_info[key]  # ...this line never executes!
    except KeyError:
        return False
    return True
```

While normal operations keep both dictionaries in sync, this bug violates the defensive programming principle and could lead to:
1. Memory leak as orphaned entries accumulate in `_expire_info`
2. Inconsistent state that could cause unexpected behavior in future operations
3. Violation of the cache's internal invariants

## Fix

```diff
--- a/django/core/cache/backends/locmem.py
+++ b/django/core/cache/backends/locmem.py
@@ -101,10 +101,14 @@ class LocMemCache(BaseCache):

     def _delete(self, key):
+        found = key in self._cache
         try:
             del self._cache[key]
-            del self._expire_info[key]
         except KeyError:
-            return False
-        return True
+            pass
+        try:
+            del self._expire_info[key]
+        except KeyError:
+            pass
+        return found
```

This fix ensures that both dictionaries are cleaned up independently, maintaining consistency even if they become desynchronized. The return value still indicates whether the key was found in the cache, preserving the original semantics.