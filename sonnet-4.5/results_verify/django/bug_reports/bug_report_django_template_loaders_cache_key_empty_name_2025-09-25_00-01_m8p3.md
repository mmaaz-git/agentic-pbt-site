# Bug Report: django.template.loaders.cached Cache Key Collision with Empty Template Name

**Target**: `django.template.loaders.cached.Loader.cache_key`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `cache_key` method creates cache key collisions when a template name is an empty string combined with skip origins, and another template whose name equals the hash value with no skip.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from hypothesis import given, strategies as st
from django.template.base import Origin
from django.template.loaders.cached import Loader as CachedLoader


@given(st.lists(st.text(), min_size=1))
def test_cache_key_uniqueness_with_empty_template_name(origin_names):
    loader = CachedLoader.__new__(CachedLoader)

    skip_origins = [Origin(name, "", None) for name in origin_names]

    key_empty_with_skip = loader.cache_key("", skip_origins)

    hash_value = loader.generate_hash(origin_names)
    key_hash_as_name = loader.cache_key(hash_value, None)

    assert ("", skip_origins) != (hash_value, None)
    assert key_empty_with_skip != key_hash_as_name, \
        f"Collision: cache_key('', {skip_origins}) == cache_key('{hash_value}', None)"
```

**Failing input**: `origin_names=['path1']`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.template.base import Origin
from django.template.loaders.cached import Loader as CachedLoader

loader = CachedLoader.__new__(CachedLoader)

skip_origins = [Origin("path1", "", None)]
key1 = loader.cache_key("", skip_origins)
print(f"cache_key('', skip) = {key1}")

hash_value = loader.generate_hash(["path1"])
key2 = loader.cache_key(hash_value, None)
print(f"cache_key('{hash_value}', None) = {key2}")

print(f"\nCollision: {key1 == key2}")
```

Output:
```
cache_key('', skip) = 074aeb9c5551d3b52d26cf3d6568599adbff99f1
cache_key('074aeb9c5551d3b52d26cf3d6568599adbff99f1', None) = 074aeb9c5551d3b52d26cf3d6568599adbff99f1

Collision: True
```

## Why This Is A Bug

The `cache_key` method at line 93 of `cached.py` uses:
```python
return "-".join(s for s in (str(template_name), skip_prefix) if s)
```

The `if s` condition filters out falsy values. When `template_name` is an empty string `''`, `str(template_name)` evaluates to `''`, which is falsy in Python. This causes the empty template name to be filtered out of the join, leaving only the `skip_prefix` hash.

This creates a collision when:
- Template name = `''` with skip origins → cache key = hash only
- Template name = the hash value itself with no skip → cache key = same hash

While unlikely in practice, this violates the expectation that different (template_name, skip) pairs should produce different cache keys.

## Fix

```diff
--- a/django/template/loaders/cached.py
+++ b/django/template/loaders/cached.py
@@ -90,4 +90,9 @@ class Loader(BaseLoader):
             if matching:
                 skip_prefix = self.generate_hash(matching)

-        return "-".join(s for s in (str(template_name), skip_prefix) if s)
+        parts = []
+        if template_name or template_name == 0:
+            parts.append(str(template_name))
+        if skip_prefix:
+            parts.append(skip_prefix)
+        return "-".join(parts)
```

Alternatively, use a more explicit condition:
```python
return "-".join(s for s in (str(template_name), skip_prefix) if s is not None and s != "")
```

Or always include the template name, even if empty:
```python
parts = [str(template_name)]
if skip_prefix:
    parts.append(skip_prefix)
return "-".join(parts)
```