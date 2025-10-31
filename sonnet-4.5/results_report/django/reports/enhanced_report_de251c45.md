# Bug Report: Django Core Cache Template Fragment Key Collision

**Target**: `django.core.cache.utils.make_template_fragment_key`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `make_template_fragment_key` function generates identical cache keys for different `vary_on` lists when elements contain the separator character `:`, causing cache collisions where unrelated cached content could be served to wrong users.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, assume, example
from django.core.cache.utils import make_template_fragment_key

@given(
    fragment_name=st.text(),
    list1=st.lists(st.text(), min_size=1, max_size=5),
    list2=st.lists(st.text(), min_size=1, max_size=5)
)
@example(fragment_name="fragment", list1=["a:", "b"], list2=["a", ":b"])
@example(fragment_name="test", list1=["x:y", "z"], list2=["x", "y:z"])
def test_different_inputs_should_produce_different_keys(fragment_name, list1, list2):
    assume(list1 != list2)

    key1 = make_template_fragment_key(fragment_name, list1)
    key2 = make_template_fragment_key(fragment_name, list2)

    assert key1 != key2, f"Cache key collision: {list1} and {list2} produce same key"

if __name__ == "__main__":
    test_different_inputs_should_produce_different_keys()
```

<details>

<summary>
**Failing input**: `fragment_name='fragment', list1=['a:', 'b'], list2=['a', ':b']`
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/23
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_different_inputs_should_produce_different_keys FAILED      [100%]

=================================== FAILURES ===================================
_____________ test_different_inputs_should_produce_different_keys ______________
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/23/hypo.py", line 8, in test_different_inputs_should_produce_different_keys
  |     fragment_name=st.text(),
  |                ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
  |     _raise_to_user(errors, state.settings, [], " in explicit examples")
  |     ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures in explicit examples. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/23/hypo.py", line 20, in test_different_inputs_should_produce_different_keys
    |     assert key1 != key2, f"Cache key collision: {list1} and {list2} produce same key"
    | AssertionError: Cache key collision: ['a:', 'b'] and ['a', ':b'] produce same key
    | assert 'template.cache.fragment.b76cc21cba4ac805d1fc53024777d235' != 'template.cache.fragment.b76cc21cba4ac805d1fc53024777d235'
    | Falsifying explicit example: test_different_inputs_should_produce_different_keys(
    |     fragment_name='fragment',
    |     list1=['a:', 'b'],
    |     list2=['a', ':b'],
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/23/hypo.py", line 20, in test_different_inputs_should_produce_different_keys
    |     assert key1 != key2, f"Cache key collision: {list1} and {list2} produce same key"
    | AssertionError: Cache key collision: ['x:y', 'z'] and ['x', 'y:z'] produce same key
    | assert 'template.cache.test.9650df9e19633bf061a181a85d966e32' != 'template.cache.test.9650df9e19633bf061a181a85d966e32'
    | Falsifying explicit example: test_different_inputs_should_produce_different_keys(
    |     fragment_name='test',
    |     list1=['x:y', 'z'],
    |     list2=['x', 'y:z'],
    | )
    +------------------------------------
=========================== short test summary info ============================
FAILED hypo.py::test_different_inputs_should_produce_different_keys - Excepti...
============================== 1 failed in 0.14s ===============================
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.core.cache.utils import make_template_fragment_key

# Test case 1: ["a:", "b"] vs ["a", ":b"]
key1 = make_template_fragment_key("fragment", ["a:", "b"])
key2 = make_template_fragment_key("fragment", ["a", ":b"])

print("Test case 1:")
print(f"vary_on=['a:', 'b'] produces key: {key1}")
print(f"vary_on=['a', ':b'] produces key: {key2}")
print(f"Keys are equal: {key1 == key2}")
print()

# Test case 2: ["x:y", "z"] vs ["x", "y:z"]
key3 = make_template_fragment_key("test", ["x:y", "z"])
key4 = make_template_fragment_key("test", ["x", "y:z"])

print("Test case 2:")
print(f"vary_on=['x:y', 'z'] produces key: {key3}")
print(f"vary_on=['x', 'y:z'] produces key: {key4}")
print(f"Keys are equal: {key3 == key4}")
```

<details>

<summary>
Cache key collision confirmed - identical keys generated for different inputs
</summary>
```
Test case 1:
vary_on=['a:', 'b'] produces key: template.cache.fragment.b76cc21cba4ac805d1fc53024777d235
vary_on=['a', ':b'] produces key: template.cache.fragment.b76cc21cba4ac805d1fc53024777d235
Keys are equal: True

Test case 2:
vary_on=['x:y', 'z'] produces key: template.cache.test.9650df9e19633bf061a181a85d966e32
vary_on=['x', 'y:z'] produces key: template.cache.test.9650df9e19633bf061a181a85d966e32
Keys are equal: True
```
</details>

## Why This Is A Bug

This violates the fundamental contract of `make_template_fragment_key`: different `vary_on` lists should produce different cache keys. The function concatenates elements with `:` as a separator without escaping or length-prefixing, creating ambiguity. When `["a:", "b"]` is processed, it produces the byte sequence `"a::b:"` (element "a:" + separator ":" + element "b" + separator ":"). When `["a", ":b"]` is processed, it produces the exact same byte sequence `"a::b:"` (element "a" + separator ":" + element ":b" + separator ":"). Since both produce identical input to the MD5 hash, they generate identical cache keys.

The Django documentation shows using this function with user data like `make_template_fragment_key("sidebar", [username])`, implying different usernames must produce different keys. If usernames or any user-controlled data contain colons, this bug enables cache poisoning where User A could receive User B's cached content - a serious security vulnerability.

## Relevant Context

The bug is in `/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/core/cache/utils.py` lines 6-12. The function is widely used in Django's template caching system to generate unique keys for cached template fragments. The `vary_on` parameter exists specifically to create distinct cache entries for different contexts (different users, different parameters, etc.).

This is a well-known class of vulnerability in hash construction called "separator ambiguity" or "length extension attack vector". Standard solutions include:
1. Length-prefixing each element before hashing
2. Using a null byte separator (which cannot appear in Python strings)
3. Escaping the separator character in inputs

Django's own documentation at https://docs.djangoproject.com/en/stable/topics/cache/#template-fragment-caching demonstrates using this function with dynamic data, making this vulnerability especially concerning for production applications.

## Proposed Fix

```diff
def make_template_fragment_key(fragment_name, vary_on=None):
    hasher = md5(usedforsecurity=False)
    if vary_on is not None:
        for arg in vary_on:
-           hasher.update(str(arg).encode())
-           hasher.update(b":")
+           arg_bytes = str(arg).encode()
+           # Use length-prefixed encoding to prevent collisions
+           hasher.update(f"{len(arg_bytes)}:".encode())
+           hasher.update(arg_bytes)
    return TEMPLATE_FRAGMENT_KEY_TEMPLATE % (fragment_name, hasher.hexdigest())
```