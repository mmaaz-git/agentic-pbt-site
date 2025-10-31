# Bug Report: xarray.backends.lru_cache.LRUCache.__delitem__() Missing Thread Lock

**Target**: `xarray.backends.lru_cache.LRUCache.__delitem__()`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `LRUCache.__delitem__()` method does not acquire the thread lock before modifying the internal cache, violating the thread-safety guarantee stated in the class documentation.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import threading
from xarray.backends.lru_cache import LRUCache

@given(
    keys=st.lists(st.integers(min_value=0, max_value=100), min_size=10, max_size=20)
)
def test_lru_cache_delitem_thread_safety(keys):
    cache = LRUCache(maxsize=50)
    for k in keys:
        cache[k] = f"value_{k}"

    errors = []
    def delete_keys():
        for k in keys:
            try:
                if k in cache:
                    del cache[k]
            except Exception as e:
                errors.append(e)

    def set_keys():
        for k in keys:
            cache[k] = f"new_value_{k}"

    threads = [threading.Thread(target=delete_keys) for _ in range(5)]
    threads.extend([threading.Thread(target=set_keys) for _ in range(5)])

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Thread-safety violations: {errors}"
```

**Failing input**: Any concurrent access pattern with deletions and modifications

## Reproducing the Bug

```python
import threading
from xarray.backends.lru_cache import LRUCache

cache = LRUCache(maxsize=10)
cache[1] = "value1"
cache[2] = "value2"

errors = []

def delete_items():
    for i in range(100):
        try:
            if 1 in cache:
                del cache[1]
        except Exception as e:
            errors.append(e)

def set_items():
    for i in range(100):
        cache[1] = f"value_{i}"

threads = [threading.Thread(target=delete_items) for _ in range(3)]
threads.extend([threading.Thread(target=set_items) for _ in range(3)])

for t in threads:
    t.start()
for t in threads:
    t.join()

if errors:
    print(f"Race conditions detected: {errors}")
```

## Why This Is A Bug

The `LRUCache` class documentation on line 13 states: "Thread-safe LRUCache based on an OrderedDict." However, the `__delitem__` method on line 81-82 of `xarray/backends/lru_cache.py` does not acquire `self._lock` before modifying `self._cache`.

This is inconsistent with `__getitem__` (line 53-58) and `__setitem__` (line 67-79), both of which correctly use `with self._lock:` to ensure thread-safety. Without the lock, concurrent deletions and modifications can cause race conditions, leading to corruption of the internal OrderedDict or unexpected exceptions.

## Fix

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