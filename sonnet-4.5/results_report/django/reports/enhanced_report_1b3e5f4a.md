# Bug Report: django.core.cache.backends.locmem - Cache Exceeds MAX_ENTRIES When CULL_FREQUENCY > MAX_ENTRIES

**Target**: `django.core.cache.backends.locmem.LocMemCache`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The LocMemCache backend violates its MAX_ENTRIES constraint when CULL_FREQUENCY is greater than the current cache size, allowing the cache to grow beyond its configured maximum and potentially causing unbounded memory growth.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from hypothesis import given, settings, strategies as st
from django.core.cache.backends.locmem import LocMemCache


def make_cache(max_entries=300, cull_frequency=3):
    return LocMemCache('test_edge', {
        'TIMEOUT': 300,
        'OPTIONS': {
            'MAX_ENTRIES': max_entries,
            'CULL_FREQUENCY': cull_frequency,
        },
        'KEY_PREFIX': 'test',
        'VERSION': 1,
    })


@given(st.integers(min_value=1, max_value=30), st.integers(min_value=1, max_value=10))
@settings(max_examples=300)
def test_exact_max_entries_boundary(max_entries, num_additional):
    cache = make_cache(max_entries=max_entries, cull_frequency=3)
    cache.clear()

    for i in range(max_entries + num_additional):
        cache.set(f"key_{i:04d}", i)

    final_size = len(cache._cache)
    assert final_size <= max_entries, \
        f"Cache size {final_size} exceeds max_entries {max_entries}"


if __name__ == "__main__":
    test_exact_max_entries_boundary()
```

<details>

<summary>
**Failing input**: `max_entries=1, num_additional=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/23/hypo.py", line 35, in <module>
    test_exact_max_entries_boundary()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/23/hypo.py", line 21, in test_exact_max_entries_boundary
    @settings(max_examples=300)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/23/hypo.py", line 30, in test_exact_max_entries_boundary
    assert final_size <= max_entries, \
           ^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Cache size 2 exceeds max_entries 1
Falsifying example: test_exact_max_entries_boundary(
    max_entries=1,
    num_additional=1,  # or any other generated value
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/23/hypo.py:31
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.core.cache.backends.locmem import LocMemCache

cache = LocMemCache('reproduce_bug', {
    'TIMEOUT': 300,
    'OPTIONS': {
        'MAX_ENTRIES': 1,
        'CULL_FREQUENCY': 3,
    },
    'KEY_PREFIX': 'test',
    'VERSION': 1,
})
cache.clear()

print(f"Initial cache size: {len(cache._cache)}")
print(f"max_entries: {cache._max_entries}")
print(f"cull_frequency: {cache._cull_frequency}")
print()

cache.set("key1", "value1")
print(f"After setting key1: cache size = {len(cache._cache)}")

cache.set("key2", "value2")
print(f"After setting key2: cache size = {len(cache._cache)}")
print()

assert len(cache._cache) <= cache._max_entries, \
    f"BUG: Cache has {len(cache._cache)} entries, exceeding max_entries={cache._max_entries}"
```

<details>

<summary>
AssertionError: Cache grows to 2 entries when max_entries=1
</summary>
```
Initial cache size: 0
max_entries: 1
cull_frequency: 3

After setting key1: cache size = 1
After setting key2: cache size = 2

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/23/repo.py", line 29, in <module>
    assert len(cache._cache) <= cache._max_entries, \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: BUG: Cache has 2 entries, exceeding max_entries=1
```
</details>

## Why This Is A Bug

This bug violates Django's documented behavior for the MAX_ENTRIES parameter. According to Django's cache documentation, MAX_ENTRIES is defined as "the maximum number of entries allowed in the cache before old values are deleted." This establishes an invariant that the cache size should never exceed MAX_ENTRIES.

The bug occurs in the interaction between the `_set` method (lines 45-50) and the `_cull` method (lines 92-100) in `/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/core/cache/backends/locmem.py`:

1. When adding a new entry, `_set` checks if the cache is at capacity: `if len(self._cache) >= self._max_entries`
2. If at capacity, it calls `_cull()` to remove entries
3. `_cull()` calculates how many items to remove: `count = len(self._cache) // self._cull_frequency`
4. When `cull_frequency > len(cache)`, integer division results in 0 items to remove
5. The new entry is then added regardless, violating the MAX_ENTRIES constraint

With the failing configuration (`max_entries=1, cull_frequency=3`):
- Cache reaches capacity with 1 entry
- Culling calculation: `1 // 3 = 0` (no items removed)
- New entry is added, resulting in 2 entries when max was 1

This violates the fundamental contract that MAX_ENTRIES represents a hard limit on cache size, potentially leading to unbounded memory growth when users expect the cache to be bounded.

## Relevant Context

The bug manifests when `CULL_FREQUENCY > MAX_ENTRIES` or more generally when `CULL_FREQUENCY > current_cache_size` at the time of culling. While this configuration might seem unusual, it's a valid configuration that Django accepts without warning or validation.

Key code locations:
- Bug location: `/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/core/cache/backends/locmem.py:97`
- Default values defined in: `/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/core/cache/backends/base.py:71-81`
- Default MAX_ENTRIES: 300
- Default CULL_FREQUENCY: 3 (meaning cull 1/3 of entries)

The issue is particularly problematic because:
1. Django provides no validation or warnings for `CULL_FREQUENCY > MAX_ENTRIES`
2. The documentation clearly states MAX_ENTRIES is a maximum, not a target
3. Users relying on MAX_ENTRIES to bound memory usage may experience unexpected memory growth

## Proposed Fix

```diff
--- a/django/core/cache/backends/locmem.py
+++ b/django/core/cache/backends/locmem.py
@@ -94,7 +94,8 @@ class LocMemCache(BaseCache):
             self._cache.clear()
             self._expire_info.clear()
         else:
-            count = len(self._cache) // self._cull_frequency
+            # Ensure at least 1 item is removed when at capacity
+            count = max(1, len(self._cache) // self._cull_frequency)
             for i in range(count):
                 key, _ = self._cache.popitem()
                 del self._expire_info[key]
```