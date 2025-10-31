# Bug Report: django.core.cache.utils.make_template_fragment_key Hash Collision

**Target**: `django.core.cache.utils.make_template_fragment_key`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `make_template_fragment_key` function has a hash collision vulnerability where different `vary_on` lists can produce identical cache keys due to improper separator handling. Specifically, `['a', 'b']` and `['a:b']` generate the same cache key.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.core.cache.utils import make_template_fragment_key

@given(
    st.lists(st.text(min_size=1), min_size=2, max_size=5)
)
def test_no_separator_collision(vary_on_list):
    joined_with_separator = ':'.join(vary_on_list)

    key1 = make_template_fragment_key('test', vary_on_list)
    key2 = make_template_fragment_key('test', [joined_with_separator])

    if vary_on_list != [joined_with_separator]:
        assert key1 != key2
```

**Failing input**: `vary_on_list = ['a', 'b']` which produces the same key as `['a:b']`

## Reproducing the Bug

```python
from django.core.cache.utils import make_template_fragment_key

key1 = make_template_fragment_key('fragment', ['a', 'b'])
key2 = make_template_fragment_key('fragment', ['a:b'])

print(f"['a', 'b'] produces: {key1}")
print(f"['a:b'] produces: {key2}")
print(f"Collision: {key1 == key2}")
```

Output:
```
['a', 'b'] produces: template.cache.fragment.d6e7a7288d38d4cf78b2f82cc7f50bba
['a:b'] produces: template.cache.fragment.d6e7a7288d38d4cf78b2f82cc7f50bba
Collision: True
```

## Why This Is A Bug

The function uses `:` as a separator when hashing vary_on elements:

```python
for arg in vary_on:
    hasher.update(str(arg).encode())
    hasher.update(b":")
```

This creates a collision because:
- `['a', 'b']` → hashes `'a' + ':' + 'b' + ':'` → `'a:b:'`
- `['a:b']` → hashes `'a:b' + ':'` → `'a:b:'`

Both produce identical hash inputs. This violates the injectivity property that different vary_on values should produce different cache keys, potentially causing cache entries to be incorrectly shared or overwritten.

## Fix

Use a separator that cannot appear in the string representation of vary_on elements, or use proper length-prefixing:

```diff
--- a/django/core/cache/utils.py
+++ b/django/core/cache/utils.py
@@ -6,8 +6,9 @@ TEMPLATE_FRAGMENT_KEY_TEMPLATE = "template.cache.%s.%s"
 def make_template_fragment_key(fragment_name, vary_on=None):
     hasher = md5(usedforsecurity=False)
     if vary_on is not None:
         for arg in vary_on:
-            hasher.update(str(arg).encode())
-            hasher.update(b":")
+            arg_bytes = str(arg).encode()
+            hasher.update(len(arg_bytes).to_bytes(4, 'big'))
+            hasher.update(arg_bytes)
     return TEMPLATE_FRAGMENT_KEY_TEMPLATE % (fragment_name, hasher.hexdigest())
```

This fix uses length-prefixing to ensure that different vary_on lists always produce different hashes, preventing collisions.