# Bug Report: django.core.cache.backends.locmem MAX_ENTRIES Constraint Violation

**Target**: `django.core.cache.backends.locmem.LocMemCache`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The LocMemCache backend violates its MAX_ENTRIES constraint when MAX_ENTRIES is smaller than CULL_FREQUENCY, allowing the cache to grow beyond its configured maximum size limit.

## Property-Based Test

```python
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'test_settings')

import django
from django.conf import settings
if not settings.configured:
    settings.configure(DEBUG=True, DATABASES={})
    django.setup()

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
        assert cache_size <= max_entries, f"Cache size {cache_size} exceeds MAX_ENTRIES {max_entries}"


if __name__ == "__main__":
    print("Running Hypothesis test for LocMemCache MAX_ENTRIES constraint...")
    print("-" * 60)
    test_max_entries_never_exceeded()
    print("✓ All tests passed!")
```

<details>

<summary>
**Failing input**: `max_entries=1, cull_frequency=2`
</summary>
```
Running Hypothesis test for LocMemCache MAX_ENTRIES constraint...
------------------------------------------------------------
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/26/hypo.py", line 35, in <module>
    test_max_entries_never_exceeded()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/26/hypo.py", line 15, in test_max_entries_never_exceeded
    @given(

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/26/hypo.py", line 29, in test_max_entries_never_exceeded
    assert cache_size <= max_entries, f"Cache size {cache_size} exceeds MAX_ENTRIES {max_entries}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Cache size 2 exceeds MAX_ENTRIES 1
Falsifying example: test_max_entries_never_exceeded(
    max_entries=1,
    cull_frequency=2,  # or any other generated value
)
```
</details>

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

# Create a cache with MAX_ENTRIES=2 and CULL_FREQUENCY=3
cache = LocMemCache('test', {
    'OPTIONS': {'MAX_ENTRIES': 2, 'CULL_FREQUENCY': 3}
})
cache.clear()

print(f"Configuration: MAX_ENTRIES={cache._max_entries}, CULL_FREQUENCY={cache._cull_frequency}")
print("-" * 60)

# Add 3 items and observe the cache size after each addition
for i in range(3):
    cache.set(f'key_{i}', f'value_{i}')
    cache_size = len(cache._cache)
    print(f"After adding key_{i}: cache size = {cache_size}")
    if cache_size > cache._max_entries:
        print(f"  ⚠️  Cache size ({cache_size}) exceeds MAX_ENTRIES ({cache._max_entries})!")

print("-" * 60)
print(f"Final state:")
print(f"  - Cache has {len(cache._cache)} entries")
print(f"  - MAX_ENTRIES is {cache._max_entries}")
print(f"  - Cache contents: {list(cache._cache.keys())}")

if len(cache._cache) > cache._max_entries:
    print(f"\n❌ BUG CONFIRMED: Cache exceeded MAX_ENTRIES limit!")
else:
    print(f"\n✓ Cache respects MAX_ENTRIES limit")
```

<details>

<summary>
Cache grows beyond MAX_ENTRIES limit
</summary>
```
Configuration: MAX_ENTRIES=2, CULL_FREQUENCY=3
------------------------------------------------------------
After adding key_0: cache size = 1
After adding key_1: cache size = 2
After adding key_2: cache size = 3
  ⚠️  Cache size (3) exceeds MAX_ENTRIES (2)!
------------------------------------------------------------
Final state:
  - Cache has 3 entries
  - MAX_ENTRIES is 2
  - Cache contents: [':1:key_2', ':1:key_1', ':1:key_0']

❌ BUG CONFIRMED: Cache exceeded MAX_ENTRIES limit!
```
</details>

## Why This Is A Bug

This bug violates the fundamental contract of MAX_ENTRIES, which is documented as "the maximum number of entries allowed in the cache before old values are deleted." The word "maximum" unambiguously indicates a hard upper limit that should never be exceeded.

The bug occurs due to a mathematical error in the culling logic at line 97 of `locmem.py`:

```python
def _cull(self):
    if self._cull_frequency == 0:
        self._cache.clear()
        self._expire_info.clear()
    else:
        count = len(self._cache) // self._cull_frequency  # BUG: Integer division can return 0
        for i in range(count):
            key, _ = self._cache.popitem()
            del self._expire_info[key]
```

When `len(self._cache) < self._cull_frequency`, the integer division returns 0, causing the for loop to execute zero times and no items to be culled. The `_set` method then proceeds to add a new item, violating the MAX_ENTRIES constraint.

This violates user expectations - a developer configuring MAX_ENTRIES=2 would reasonably expect the cache to never contain more than 2 items, regardless of other configuration parameters.

## Relevant Context

- **Default Configuration**: The default values (MAX_ENTRIES=300, CULL_FREQUENCY=3) don't trigger this bug, which explains why it may have gone unnoticed.
- **Triggering Condition**: The bug occurs specifically when `MAX_ENTRIES < CULL_FREQUENCY`.
- **Code Location**: The bug is in `/django/core/cache/backends/locmem.py`, line 97.
- **Django Documentation**: https://docs.djangoproject.com/en/stable/topics/cache/#local-memory-caching
- **Similar Implementation**: The FileBasedCache has a similar culling mechanism but avoids this bug by checking cache size before adding new items (line 108).

## Proposed Fix

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
