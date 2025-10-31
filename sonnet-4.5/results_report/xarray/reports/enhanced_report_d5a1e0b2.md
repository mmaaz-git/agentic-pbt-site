# Bug Report: xarray.backends.lru_cache.LRUCache.__delitem__ Missing Thread Lock

**Target**: `xarray.backends.lru_cache.LRUCache.__delitem__`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `__delitem__` method in the LRUCache class fails to acquire the thread lock before modifying `self._cache`, violating the class's documented thread-safety guarantee and creating a race condition.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

import threading
from hypothesis import given, strategies as st
from xarray.backends.lru_cache import LRUCache

@given(st.integers(min_value=10, max_value=100))
def test_lrucache_delitem_thread_safe(n_threads):
    cache = LRUCache(maxsize=100)

    for i in range(10):
        cache[i] = f"value_{i}"

    errors = []

    def delete_all():
        for i in range(10):
            try:
                del cache[i]
            except KeyError:
                pass
            except Exception as e:
                errors.append(e)

    threads = [threading.Thread(target=delete_all) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread-safety violated: {errors}"

if __name__ == "__main__":
    test_lrucache_delitem_thread_safe()
```

<details>

<summary>
**Failing input**: Race conditions are non-deterministic and may not manifest consistently
</summary>
```
Test completed without errors - race condition may not manifest consistently
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

import threading
from xarray.backends.lru_cache import LRUCache

# Create an LRUCache instance
cache = LRUCache(maxsize=10)

# Add some items to the cache
for i in range(5):
    cache[i] = f"value_{i}"

# Function to delete items from the cache
def delete_items():
    for i in range(5):
        try:
            del cache[i]
        except KeyError:
            pass  # Item already deleted by another thread

# Create multiple threads that will try to delete the same items
threads = []
for _ in range(10):
    t = threading.Thread(target=delete_items)
    threads.append(t)
    t.start()

# Wait for all threads to complete
for t in threads:
    t.join()

print("Test completed - race condition may have occurred silently")

# Verify all items were deleted
remaining_items = list(cache.keys())
print(f"Remaining items in cache: {remaining_items}")
print(f"Cache size: {len(cache)}")
```

<details>

<summary>
Race condition did not manifest visibly but may have occurred internally
</summary>
```
Test completed - race condition may have occurred silently
Remaining items in cache: []
Cache size: 0
```
</details>

## Why This Is A Bug

This violates the explicit thread-safety guarantee documented in the class docstring. The LRUCache class states on line 13: "Thread-safe LRUCache based on an OrderedDict." This creates a contract that ALL operations on the cache should be thread-safe.

Code inspection reveals the inconsistency:

1. **The class maintains a lock** (`self._lock = threading.RLock()` on line 50)
2. **Other mutating methods properly acquire the lock:**
   - `__getitem__` (lines 55-58): Uses `with self._lock` before accessing `self._cache`
   - `__setitem__` (lines 68-79): Uses `with self._lock` before modifying `self._cache`
   - `maxsize.setter` (lines 102-104): Uses `with self._lock` before calling `_enforce_size_limit`
3. **However, `__delitem__` (lines 81-82) does NOT acquire the lock:**
   ```python
   def __delitem__(self, key: K) -> None:
       del self._cache[key]  # Direct modification without lock!
   ```

This creates several problems:
- **Race condition**: Multiple threads can simultaneously modify the internal OrderedDict
- **Data corruption**: The OrderedDict's internal state can become corrupted when modified concurrently
- **Affects inherited methods**: Methods like `pop()` and `popitem()` from MutableMapping call `__delitem__` internally, propagating the thread-safety violation
- **Silent failures**: Race conditions may not cause immediate crashes but can lead to subtle bugs in production

While the race condition may not consistently manifest in testing due to timing dependencies, the code violation is clear and unambiguous.

## Relevant Context

The LRUCache class is used internally by xarray for caching file handles and other resources in multi-threaded environments. The class inherits from `MutableMapping[K, V]` which provides several methods that internally use `__delitem__`:
- `pop(key)` - Gets item then calls `__delitem__`
- `popitem()` - Iterates and calls `__delitem__`
- `clear()` - Repeatedly calls `popitem()`

Source code location: `/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages/xarray/backends/lru_cache.py`

The missing lock is particularly concerning because:
1. The infrastructure for thread-safety already exists (the lock is created and used elsewhere)
2. This appears to be an oversight rather than intentional design
3. The fix is trivial but the consequences of the bug could be serious in production systems

## Proposed Fix

```diff
--- a/xarray/backends/lru_cache.py
+++ b/xarray/backends/lru_cache.py
@@ -79,7 +79,8 @@ class LRUCache(MutableMapping[K, V]):
                 self._on_evict(key, value)

     def __delitem__(self, key: K) -> None:
-        del self._cache[key]
+        with self._lock:
+            del self._cache[key]

     def __iter__(self) -> Iterator[K]:
         # create a list, so accessing the cache during iteration cannot change
```