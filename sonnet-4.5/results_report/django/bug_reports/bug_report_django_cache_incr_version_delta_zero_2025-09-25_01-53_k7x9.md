# Bug Report: django.core.cache.backends incr_version Deletes Value When delta=0

**Target**: `django.core.cache.backends.base.BaseCache.incr_version`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `incr_version` method incorrectly deletes the cache value when called with `delta=0`. The method sets the value at the new version (which equals the old version when delta=0), then immediately deletes it, resulting in data loss.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.core.cache.backends.locmem import LocMemCache

@given(st.text(min_size=1), st.integers(), st.integers(min_value=-10, max_value=10))
def test_incr_version_with_delta(key, value, delta):
    cache = LocMemCache("test", {"timeout": 300})
    cache.clear()

    initial_version = 1
    cache.set(key, value, version=initial_version)

    new_version = cache.incr_version(key, delta=delta, version=initial_version)

    assert new_version == initial_version + delta

    result_new = cache.get(key, version=new_version)
    assert result_new == value, f"New version: Expected {value}, got {result_new}"

    result_old = cache.get(key, default="MISSING", version=initial_version)
    assert result_old == "MISSING", f"Old version should be deleted, got {result_old}"
```

**Failing input**: `key='0', value=0, delta=0`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.core.cache.backends.locmem import LocMemCache

cache = LocMemCache("test", {"timeout": 300})
cache.clear()

cache.set("mykey", 42, version=1)
new_version = cache.incr_version("mykey", delta=0, version=1)

result = cache.get("mykey", version=new_version)

print(f"Expected: 42")
print(f"Actual: {result}")
```

## Why This Is A Bug

Looking at the `incr_version` implementation in `/django/core/cache/backends/base.py` lines 346-360:

```python
def incr_version(self, key, delta=1, version=None):
    if version is None:
        version = self.version

    value = self.get(key, self._missing_key, version=version)
    if value is self._missing_key:
        raise ValueError("Key '%s' not found" % key)

    self.set(key, value, version=version + delta)  # Line 358
    self.delete(key, version=version)              # Line 359
    return version + delta
```

When `delta=0`:
1. Line 358 sets the value at `version + 0 = version`
2. Line 359 deletes the value at `version`

Since both operations target the same version, the value is set and then immediately deleted, resulting in data loss.

While `delta=0` might seem like an edge case, the method accepts it as a valid parameter (no validation), and users might reasonably expect it to be a no-op or to preserve the value.

## Fix

```diff
--- a/django/core/cache/backends/base.py
+++ b/django/core/cache/backends/base.py
@@ -350,6 +350,10 @@ class BaseCache:
         if version is None:
             version = self.version

+        # Handle delta=0 case to avoid set/delete at same version
+        if delta == 0:
+            return version
+
         value = self.get(key, self._missing_key, version=version)
         if value is self._missing_key:
             raise ValueError("Key '%s' not found" % key)
```

Alternatively, only delete if versions differ:

```diff
--- a/django/core/cache/backends/base.py
+++ b/django/core/cache/backends/base.py
@@ -356,7 +356,8 @@ class BaseCache:
             raise ValueError("Key '%s' not found" % key)

         self.set(key, value, version=version + delta)
-        self.delete(key, version=version)
+        if delta != 0:
+            self.delete(key, version=version)
         return version + delta
```