# Bug Report: Django Core Cache Template Fragment Key Collision

**Target**: `django.core.cache.utils.make_template_fragment_key`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `make_template_fragment_key` function produces identical cache keys for different `vary_on` arguments when those arguments contain the separator character `:`, causing cache collisions and potential security vulnerabilities.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
from django.core.cache.utils import make_template_fragment_key

@given(
    fragment_name=st.text(),
    list1=st.lists(st.text(), min_size=1, max_size=5),
    list2=st.lists(st.text(), min_size=1, max_size=5)
)
def test_different_inputs_should_produce_different_keys(fragment_name, list1, list2):
    assume(list1 != list2)

    key1 = make_template_fragment_key(fragment_name, list1)
    key2 = make_template_fragment_key(fragment_name, list2)

    assert key1 != key2, f"Cache key collision: {list1} and {list2} produce same key"
```

**Failing inputs**:
- `vary_on_1 = ["a:", "b"]` and `vary_on_2 = ["a", ":b"]`
- `vary_on_1 = ["x:y", "z"]` and `vary_on_2 = ["x", "y:z"]`

## Reproducing the Bug

```python
from django.core.cache.utils import make_template_fragment_key

key1 = make_template_fragment_key("fragment", ["a:", "b"])
key2 = make_template_fragment_key("fragment", ["a", ":b"])

print(f"Key 1: {key1}")
print(f"Key 2: {key2}")
print(f"Keys are equal: {key1 == key2}")

key3 = make_template_fragment_key("test", ["x:y", "z"])
key4 = make_template_fragment_key("test", ["x", "y:z"])

print(f"\nKey 3: {key3}")
print(f"Key 4: {key4}")
print(f"Keys are equal: {key3 == key4}")
```

## Why This Is A Bug

The function uses `:` as a separator between hashed elements (line 11 of `django/core/cache/utils.py`):

```python
def make_template_fragment_key(fragment_name, vary_on=None):
    hasher = md5(usedforsecurity=False)
    if vary_on is not None:
        for arg in vary_on:
            hasher.update(str(arg).encode())
            hasher.update(b":")  # Problem: ambiguous separator
    return TEMPLATE_FRAGMENT_KEY_TEMPLATE % (fragment_name, hasher.hexdigest())
```

The issue is that the separator `:` is not escaped or protected, so different lists can produce the same hash input:

- `["a:", "b"]` → hash input: `"a::" + "b:"` → `"a::b:"`
- `["a", ":b"]` → hash input: `"a:" + ":b:"` → `"a::b:"`

This violates the fundamental property that **different inputs should produce different cache keys**. The collision can lead to:

1. **Cache poisoning**: User A's cached fragment could be served to User B
2. **Security issues**: Sensitive data leakage between different contexts
3. **Correctness bugs**: Wrong content being displayed from cache

## Fix

Use a length-prefixed encoding or a more robust separator scheme:

```diff
def make_template_fragment_key(fragment_name, vary_on=None):
    hasher = md5(usedforsecurity=False)
    if vary_on is not None:
        for arg in vary_on:
-           hasher.update(str(arg).encode())
-           hasher.update(b":")
+           arg_bytes = str(arg).encode()
+           # Use length prefix to prevent ambiguity
+           hasher.update(f"{len(arg_bytes)}:".encode())
+           hasher.update(arg_bytes)
    return TEMPLATE_FRAGMENT_KEY_TEMPLATE % (fragment_name, hasher.hexdigest())
```

Alternative fix using null byte separator (more common pattern):

```diff
def make_template_fragment_key(fragment_name, vary_on=None):
    hasher = md5(usedforsecurity=False)
    if vary_on is not None:
        for arg in vary_on:
            hasher.update(str(arg).encode())
-           hasher.update(b":")
+           hasher.update(b"\x00")  # Null byte is unambiguous
    return TEMPLATE_FRAGMENT_KEY_TEMPLATE % (fragment_name, hasher.hexdigest())
```