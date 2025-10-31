# Bug Report: xarray.backends.lru_cache.LRUCache.__delitem__ Not Thread-Safe

**Target**: `xarray.backends.lru_cache.LRUCache.__delitem__`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `LRUCache.__delitem__` method modifies the internal `_cache` dictionary without acquiring the thread lock, violating the class's documented thread-safety guarantee. This can cause race conditions when deletion occurs concurrently with other cache operations.

## Property-Based Test

```python
import threading
from hypothesis import given, strategies as st, settings
from xarray.backends.lru_cache import LRUCache


@given(
    st.lists(st.integers(min_value=0, max_value=99), min_size=10, max_size=100),
    st.lists(st.integers(min_value=0, max_value=99), min_size=10, max_size=100),
)
@settings(max_examples=100)
def test_lrucache_concurrent_delete_and_get(keys_to_delete, keys_to_get):
    cache = LRUCache(maxsize=100)

    for i in range(100):
        cache[i] = f"value_{i}"

    errors = []

    def deleter():
        try:
            for key in keys_to_delete:
                if key in cache:
                    del cache[key]
        except Exception as e:
            errors.append(e)

    def getter():
        try:
            for key in keys_to_get:
                if key in cache:
                    _ = cache[key]
        except Exception as e:
            errors.append(e)

    t1 = threading.Thread(target=deleter)
    t2 = threading.Thread(target=getter)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert len(errors) == 0, f"Race condition detected: {errors}"
```

**Failing input**: Various combinations of concurrent deletes and gets can trigger `RuntimeError: dictionary changed size during iteration` or `KeyError`.

## Reproducing the Bug

```python
import threading
from xarray.backends.lru_cache import LRUCache

cache = LRUCache(maxsize=100)
for i in range(100):
    cache[i] = f"value_{i}"

def deleter():
    for _ in range(1000):
        for i in range(50):
            if i in cache:
                del cache[i]

def reader():
    for _ in range(1000):
        for i in range(50):
            if i in cache:
                _ = cache[i]

t1 = threading.Thread(target=deleter)
t2 = threading.Thread(target=reader)
t1.start()
t2.start()
t1.join()
t2.join()
```

Running this may produce `RuntimeError`, `KeyError`, or incorrect behavior due to race conditions.

## Why This Is A Bug

The class docstring explicitly states: "Thread-safe LRUCache based on an OrderedDict."

All other mutating methods (`__getitem__`, `__setitem__`, `maxsize.setter`) acquire `self._lock` before modifying `self._cache`. Only `__delitem__` fails to do so, violating the thread-safety contract.

According to Python's OrderedDict documentation, dictionary operations are not thread-safe without external synchronization. Concurrent modification can lead to:
- `RuntimeError: dictionary changed size during iteration`
- `KeyError` exceptions
- Internal corruption of the OrderedDict structure

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