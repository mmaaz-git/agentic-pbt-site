# Bug Report: Django Cache incr_version Deletes Value When delta=0

**Target**: `django.core.cache.backends.base.BaseCache.incr_version`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `incr_version` method in Django's cache backend incorrectly deletes cached values when called with `delta=0`. Instead of preserving the value at the same version (a logical no-op), the method deletes the data entirely due to a logic flaw in its implementation.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Hypothesis-based property test for Django cache incr_version method.

This test verifies that incr_version correctly preserves the cached value
at the new version for all valid delta values, including delta=0.
"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

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

if __name__ == "__main__":
    # Run the test
    test_incr_version_with_delta()
```

<details>

<summary>
**Failing input**: `key='0', value=0, delta=0`
</summary>
```
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':1:\x90d\x15ô\U00095966'
  warnings.warn(warning, CacheKeyWarning)
[... more warnings omitted for brevity ...]
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/33/hypo.py", line 35, in <module>
    test_incr_version_with_delta()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/33/hypo.py", line 16, in test_incr_version_with_delta
    def test_incr_version_with_delta(key, value, delta):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/33/hypo.py", line 28, in test_incr_version_with_delta
    assert result_new == value, f"New version: Expected {value}, got {result_new}"
           ^^^^^^^^^^^^^^^^^^^
AssertionError: New version: Expected 0, got None
Falsifying example: test_incr_version_with_delta(
    # The test sometimes passed when commented parts were varied together.
    key='0',  # or any other generated value
    value=0,  # or any other generated value
    delta=0,
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Demonstration of Django cache incr_version bug when delta=0.

This script shows that calling incr_version with delta=0 incorrectly
deletes the cached value instead of preserving it at the same version.
"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.core.cache.backends.locmem import LocMemCache

print("=== Django Cache incr_version Bug Demonstration ===")
print("Testing behavior when incrementing cache version by delta=0")
print()

# Initialize cache
cache = LocMemCache("test", {"timeout": 300})
cache.clear()

# Test case 1: Basic delta=0 scenario
print("Test 1: Basic delta=0 scenario")
print("-" * 40)
cache.set("mykey", 42, version=1)
print(f"Initial setup: set('mykey', 42, version=1)")
print(f"Value at version 1 before incr_version: {cache.get('mykey', version=1)}")

new_version = cache.incr_version("mykey", delta=0, version=1)
print(f"Called incr_version('mykey', delta=0, version=1)")
print(f"Returned new_version: {new_version}")

result = cache.get("mykey", version=new_version)
print(f"Value at version {new_version} after incr_version: {result}")
print(f"Expected: 42")
print(f"Actual: {result}")
print(f"BUG: Value was deleted! (got None instead of 42)")
print()

# Test case 2: Compare with non-zero delta values
print("Test 2: Comparing different delta values")
print("-" * 40)

test_cases = [
    ("delta=-1", -1),
    ("delta=0", 0),
    ("delta=1", 1),
    ("delta=2", 2),
]

for label, delta in test_cases:
    cache.clear()
    cache.set("testkey", 100, version=5)

    try:
        new_ver = cache.incr_version("testkey", delta=delta, version=5)
        val_at_new = cache.get("testkey", version=new_ver)
        val_at_old = cache.get("testkey", version=5)

        print(f"{label:10} -> new_version={new_ver}, value={val_at_new}, old_version_value={val_at_old}")

        if delta == 0 and val_at_new is None:
            print(f"           ^^ BUG: Value deleted when delta=0!")
    except Exception as e:
        print(f"{label:10} -> Error: {e}")

print()
print("=== Summary ===")
print("When delta=0, incr_version incorrectly deletes the cached value.")
print("This happens because the implementation:")
print("1. Sets the value at version + 0 (same version)")
print("2. Then deletes the value at the original version")
print("3. Since both operations target the same version, the value is lost")
```

<details>

<summary>
Output shows value is deleted when delta=0
</summary>
```
=== Django Cache incr_version Bug Demonstration ===
Testing behavior when incrementing cache version by delta=0

Test 1: Basic delta=0 scenario
----------------------------------------
Initial setup: set('mykey', 42, version=1)
Value at version 1 before incr_version: 42
Called incr_version('mykey', delta=0, version=1)
Returned new_version: 1
Value at version 1 after incr_version: None
Expected: 42
Actual: None
BUG: Value was deleted! (got None instead of 42)

Test 2: Comparing different delta values
----------------------------------------
delta=-1   -> new_version=4, value=100, old_version_value=None
delta=0    -> new_version=5, value=None, old_version_value=None
           ^^ BUG: Value deleted when delta=0!
delta=1    -> new_version=6, value=100, old_version_value=None
delta=2    -> new_version=7, value=100, old_version_value=None

=== Summary ===
When delta=0, incr_version incorrectly deletes the cached value.
This happens because the implementation:
1. Sets the value at version + 0 (same version)
2. Then deletes the value at the original version
3. Since both operations target the same version, the value is lost
```
</details>

## Why This Is A Bug

The `incr_version` method is designed to move a cached value from one version to another by incrementing the version number. However, when `delta=0`, the method exhibits incorrect behavior that violates reasonable expectations:

1. **Silent Data Loss**: The method completely deletes the cached value when `delta=0` is used, which is unexpected and dangerous behavior. No reasonable interpretation of "increment version by 0" would include "delete the data."

2. **Flawed Implementation Logic**: Looking at the source code in `/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/core/cache/backends/base.py`, lines 346-360:
   ```python
   def incr_version(self, key, delta=1, version=None):
       """
       Add delta to the cache version for the supplied key. Return the new
       version.
       """
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
   - Line 358 sets the value at `version + 0`, which equals `version`
   - Line 359 immediately deletes the value at `version`
   - Since both operations target the exact same version key, the value is first set and then immediately deleted

3. **No Input Validation**: The method accepts `delta=0` without any validation or error. If this were truly an invalid input, the method should explicitly reject it. The parameter name "delta" (a mathematical term that includes 0 as a valid value) suggests all integer values should be supported.

4. **Related Method Support**: The `decr_version` method is implemented as `incr_version(key, -delta, version)`, showing that the method is designed to handle various delta values including negative ones. This suggests `delta=0` should also be a valid input.

5. **Documentation Gap**: The method's docstring states it will "Add delta to the cache version for the supplied key" but doesn't specify behavior for `delta=0` or indicate it's unsupported.

## Relevant Context

- **Django Version**: This bug affects Django's core cache backend implementation in the base class that all cache backends inherit from.
- **Impact Scope**: Any cache backend that doesn't override `incr_version` will exhibit this bug (LocMemCache, FileBasedCache, etc.)
- **Use Case**: While `delta=0` might seem like an edge case, it could reasonably occur in dynamic code where the delta is computed and might sometimes be 0, or when a developer expects it to work as a no-op
- **Workaround**: Developers can avoid this bug by checking if `delta==0` before calling `incr_version` and skipping the call entirely
- **Related Django Documentation**: [Django Cache Framework](https://docs.djangoproject.com/en/stable/topics/cache/)

## Proposed Fix

```diff
--- a/django/core/cache/backends/base.py
+++ b/django/core/cache/backends/base.py
@@ -350,6 +350,10 @@ class BaseCache:
         if version is None:
             version = self.version

+        # Handle delta=0 case - should be a no-op that preserves the value
+        if delta == 0:
+            return version
+
         value = self.get(key, self._missing_key, version=version)
         if value is self._missing_key:
             raise ValueError("Key '%s' not found" % key)
```