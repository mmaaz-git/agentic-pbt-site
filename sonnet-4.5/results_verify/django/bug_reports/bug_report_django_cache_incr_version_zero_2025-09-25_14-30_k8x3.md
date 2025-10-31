# Bug Report: django.core.cache.backends.base.BaseCache.incr_version with delta=0

**Target**: `django.core.cache.backends.base.BaseCache.incr_version`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Calling `incr_version(key, delta=0)` silently deletes the cached value instead of being a no-op, causing unexpected data loss.

## Property-Based Test

```python
import sys
from hypothesis import given, strategies as st

sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.core.cache.backends.locmem import LocMemCache


@given(st.text(min_size=1), st.integers(), st.integers(min_value=1, max_value=100))
def test_incr_version_zero_delta_deletes_key(key, value, version):
    """Property: incr_version with delta=0 should not delete the key."""
    cache = LocMemCache(f"test_{version}", {"timeout": 300})
    cache.clear()

    cache.set(key, value, version=version)
    assert cache.has_key(key, version=version), "Key should exist before incr_version"

    new_version = cache.incr_version(key, delta=0, version=version)
    assert new_version == version, f"Version should remain {version}, got {new_version}"

    assert cache.has_key(key, version=version), \
        f"Key should still exist at version {version} after incr_version with delta=0"
```

**Failing input**: `key='0', value=0, version=1`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.core.cache.backends.locmem import LocMemCache

cache = LocMemCache("test", {"timeout": 300})
cache.clear()

cache.set("mykey", "myvalue", version=1)
print(f"Before: has_key('mykey', version=1) = {cache.has_key('mykey', version=1)}")

cache.incr_version("mykey", delta=0, version=1)
print(f"After: has_key('mykey', version=1) = {cache.has_key('mykey', version=1)}")
```

Output:
```
Before: has_key('mykey', version=1) = True
After: has_key('mykey', version=1) = False
```

## Why This Is A Bug

The `incr_version` method is designed to move a cached value from one version to another. When `delta=0`, the source and destination versions are identical. The implementation:

1. Gets the value from version `v`
2. Sets the value at version `v + delta` (which is `v` when delta=0)
3. Deletes the value from version `v`

Since steps 2 and 3 operate on the same version when delta=0, the value is overwritten and then immediately deleted, resulting in data loss. This violates the reasonable expectation that either:
- `delta=0` should be a no-op (preserve the key), or
- The function should reject `delta=0` with a clear error

The same bug affects `decr_version(key, delta=0)` since it calls `incr_version(key, -delta)`.

## Fix

```diff
--- a/django/core/cache/backends/base.py
+++ b/django/core/cache/backends/base.py
@@ -348,6 +348,9 @@ class BaseCache:
         Add delta to the cache version for the supplied key. Return the new
         version.
         """
+        if delta == 0:
+            # No-op: return current version without modifying cache
+            return self.version if version is None else version
         if version is None:
             version = self.version

@@ -363,6 +366,9 @@ class BaseCache:
     async def aincr_version(self, key, delta=1, version=None):
         """See incr_version()."""
+        if delta == 0:
+            # No-op: return current version without modifying cache
+            return self.version if version is None else version
         if version is None:
             version = self.version
```