# Bug Report: xarray.backends.lru_cache.LRUCache.__delitem__ Not Thread-Safe

**Target**: `xarray.backends.lru_cache.LRUCache.__delitem__`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `LRUCache.__delitem__` method does not acquire the thread lock before modifying `self._cache`, violating the class's thread-safety guarantee stated in its docstring.

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
```

**Failing input**: Can fail with race conditions when `n_threads > 1`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

import threading
from xarray.backends.lru_cache import LRUCache

cache = LRUCache(maxsize=10)
cache[1] = "test"

def delete_item():
    try:
        del cache[1]
    except KeyError:
        pass

threads = [threading.Thread(target=delete_item) for _ in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print("Test completed - may have race condition")
```

## Why This Is A Bug

The class docstring states (line 13 in lru_cache.py):
> "Thread-safe LRUCache based on an OrderedDict."

And further states (line 15-16):
> "All dict operations (__getitem__, __setitem__, __contains__) update the priority of the relevant key and take O(1) time."

The implementation of `__getitem__` (lines 53-58) and `__setitem__` (lines 67-79) both properly acquire `self._lock`:

```python
def __getitem__(self, key: K) -> V:
    with self._lock:
        value = self._cache[key]
        self._cache.move_to_end(key)
        return value

def __setitem__(self, key: K, value: V) -> None:
    with self._lock:
        # ... operations on self._cache
```

However, `__delitem__` (lines 81-82) does NOT acquire the lock:

```python
def __delitem__(self, key: K) -> None:
    del self._cache[key]  # BUG: No lock acquired!
```

This creates a race condition where:
1. Thread A is in `__setitem__`, holding the lock and modifying `self._cache`
2. Thread B calls `__delitem__` without acquiring the lock
3. Both threads modify `self._cache` simultaneously, violating thread safety

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