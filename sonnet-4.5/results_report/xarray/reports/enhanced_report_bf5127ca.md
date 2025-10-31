# Bug Report: xarray.backends.lru_cache.LRUCache.__delitem__() Missing Thread Lock

**Target**: `xarray.backends.lru_cache.LRUCache.__delitem__()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `LRUCache.__delitem__()` method does not acquire the thread lock before modifying the internal cache, violating the thread-safety guarantee explicitly stated in the class documentation.

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

<details>

<summary>
**Failing input**: Test passes without detecting race condition
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/15
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_lru_cache_delitem_thread_safety PASSED                     [100%]

============================== 1 passed in 0.55s ===============================
```
</details>

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
else:
    print("No race conditions detected (this doesn't mean the bug doesn't exist)")
```

<details>

<summary>
No runtime error but code inspection reveals missing synchronization
</summary>
```
No race conditions detected (this doesn't mean the bug doesn't exist)
```
</details>

## Why This Is A Bug

The `LRUCache` class documentation on line 13 of `xarray/backends/lru_cache.py` explicitly states: "Thread-safe LRUCache based on an OrderedDict." This establishes a clear contract that ALL operations on the cache should be thread-safe.

The class correctly implements thread-safety for most operations:
- `__getitem__` (lines 53-58): Properly acquires lock with `with self._lock:`
- `__setitem__` (lines 67-79): Properly acquires lock with `with self._lock:`
- `maxsize.setter` (lines 98-104): Properly acquires lock with `with self._lock:`

However, `__delitem__` (lines 81-82) violates this contract:
```python
def __delitem__(self, key: K) -> None:
    del self._cache[key]  # No lock acquisition!
```

This creates a race condition where:
1. Thread A could be deleting a key from the OrderedDict without holding the lock
2. Thread B could simultaneously be accessing/modifying the same OrderedDict while holding the lock
3. This could lead to corrupted internal OrderedDict state, KeyErrors, or other undefined behavior

While the race condition is difficult to reproduce empirically (likely due to CPython's GIL providing some protection), the bug is clear from code inspection. The missing lock violates the documented thread-safety guarantee and could manifest in:
- Non-CPython implementations without a GIL (Jython, IronPython)
- Future CPython versions without GIL
- High-concurrency scenarios with specific timing

## Relevant Context

The `LRUCache` class inherits from `MutableMapping[K, V]` which requires implementing `__delitem__` as part of the standard dictionary interface. Since the class promises thread-safety for "all dict operations" (line 15), users reasonably expect `del cache[key]` to be thread-safe.

The class uses `threading.RLock` (initialized on line 50) for synchronization. An RLock (reentrant lock) allows the same thread to acquire it multiple times, which is appropriate for this use case since some methods may call other methods internally.

Source file location: `/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages/xarray/backends/lru_cache.py`

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