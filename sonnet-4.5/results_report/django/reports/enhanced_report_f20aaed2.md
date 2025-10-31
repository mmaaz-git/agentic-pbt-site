# Bug Report: django.core.cache.utils.make_template_fragment_key Hash Collision Due to Separator Ambiguity

**Target**: `django.core.cache.utils.make_template_fragment_key`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `make_template_fragment_key` function generates identical cache keys for different `vary_on` lists when the colon character (`:`) appears in the list elements, violating the documented guarantee that unique arguments produce unique cache entries.

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
        assert key1 != key2, f"Collision: {vary_on_list} produces same key as [{joined_with_separator}]"

# Run the test
if __name__ == "__main__":
    test_no_separator_collision()
```

<details>

<summary>
**Failing input**: `vary_on_list=['0', '0']`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/3/hypo.py", line 18, in <module>
    test_no_separator_collision()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/3/hypo.py", line 5, in test_no_separator_collision
    st.lists(st.text(min_size=1), min_size=2, max_size=5)
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/3/hypo.py", line 14, in test_no_separator_collision
    assert key1 != key2, f"Collision: {vary_on_list} produces same key as [{joined_with_separator}]"
           ^^^^^^^^^^^^
AssertionError: Collision: ['0', '0'] produces same key as [0:0]
Falsifying example: test_no_separator_collision(
    vary_on_list=['0', '0'],  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
from django.core.cache.utils import make_template_fragment_key

# Test case showing the collision
key1 = make_template_fragment_key('fragment', ['a', 'b'])
key2 = make_template_fragment_key('fragment', ['a:b'])

print(f"['a', 'b'] produces: {key1}")
print(f"['a:b'] produces: {key2}")
print(f"Collision: {key1 == key2}")

# Additional test cases showing more collisions
key3 = make_template_fragment_key('test', ['x', 'y', 'z'])
key4 = make_template_fragment_key('test', ['x:y', 'z'])
key5 = make_template_fragment_key('test', ['x', 'y:z'])
key6 = make_template_fragment_key('test', ['x:y:z'])

print(f"\n['x', 'y', 'z'] produces: {key3}")
print(f"['x:y', 'z'] produces: {key4}")
print(f"['x', 'y:z'] produces: {key5}")
print(f"['x:y:z'] produces: {key6}")
print(f"All four keys equal: {key3 == key4 == key5 == key6}")
```

<details>

<summary>
Hash collision demonstrated - multiple different vary_on lists produce identical cache keys
</summary>
```
['a', 'b'] produces: template.cache.fragment.d6e7a7288d38d4cf78b2f82cc7f50bba
['a:b'] produces: template.cache.fragment.d6e7a7288d38d4cf78b2f82cc7f50bba
Collision: True

['x', 'y', 'z'] produces: template.cache.test.9650df9e19633bf061a181a85d966e32
['x:y', 'z'] produces: template.cache.test.9650df9e19633bf061a181a85d966e32
['x', 'y:z'] produces: template.cache.test.9650df9e19633bf061a181a85d966e32
['x:y:z'] produces: template.cache.test.9650df9e19633bf061a181a85d966e32
All four keys equal: True
```
</details>

## Why This Is A Bug

This bug violates the documented contract in `/django/templatetags/cache.py` line 82 which states: "Each unique set of arguments will result in a unique cache entry". The function fails to maintain injectivity between different `vary_on` lists and their generated cache keys.

The root cause lies in the implementation at `/django/core/cache/utils.py` lines 9-11:
```python
for arg in vary_on:
    hasher.update(str(arg).encode())
    hasher.update(b":")
```

The function concatenates each element with a colon separator before hashing. This creates ambiguity:
- `['a', 'b']` → hashes the bytes: `'a' + ':' + 'b' + ':'` → `'a:b:'`
- `['a:b']` → hashes the bytes: `'a:b' + ':'` → `'a:b:'`

Both produce identical byte sequences for the MD5 hash, resulting in the same cache key. This breaks the fundamental requirement that different inputs should produce different cache keys, potentially causing:
1. Wrong cached content served to users
2. Cache entries being incorrectly overwritten
3. Security/privacy issues if user-specific content is incorrectly shared

## Relevant Context

The `make_template_fragment_key` function is used by Django's template caching system to generate unique keys for cached template fragments. The `{% cache %}` template tag (defined in `/django/templatetags/cache.py`) relies on this function to cache expensive template operations with different variations based on the `vary_on` parameters.

Example template usage where this bug could manifest:
```django
{% cache 500 sidebar request.user.username request.path %}
    ... expensive sidebar generation ...
{% endcache %}
```

If `request.user.username` is "user1" and `request.path` is "data", this would collide with a user named "user1:data" accessing the root path, serving the wrong cached sidebar.

Documentation reference: https://docs.djangoproject.com/en/stable/topics/cache/#template-fragment-caching

## Proposed Fix

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