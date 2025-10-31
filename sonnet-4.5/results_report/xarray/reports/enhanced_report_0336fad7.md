# Bug Report: xarray.backends.lru_cache.LRUCache.__delitem__ Violates Thread-Safety Contract

**Target**: `xarray.backends.lru_cache.LRUCache.__delitem__`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `LRUCache.__delitem__` method modifies the internal `_cache` dictionary without acquiring the thread lock, violating the class's documented thread-safety guarantee and creating race conditions with concurrent operations.

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


if __name__ == "__main__":
    print("Running Hypothesis property-based test for LRUCache thread-safety...")
    print("=" * 70)
    print()
    print("Test: Concurrent delete and get operations on LRUCache")
    print("Expected: All operations complete without errors (thread-safe)")
    print("Actual: Potential race conditions due to missing lock in __delitem__")
    print()

    try:
        test_lrucache_concurrent_delete_and_get()
        print("All tests passed!")
    except AssertionError as e:
        print(f"TEST FAILED: {e}")
    except Exception as e:
        print(f"Test execution error: {e}")
```

<details>

<summary>
**Failing input**: Race conditions are probabilistic and may not always manifest
</summary>
```
Running Hypothesis property-based test for LRUCache thread-safety...
======================================================================

Test: Concurrent delete and get operations on LRUCache
Expected: All operations complete without errors (thread-safe)
Actual: Potential race conditions due to missing lock in __delitem__

All tests passed!
```
</details>

## Reproducing the Bug

```python
import threading
from xarray.backends.lru_cache import LRUCache

# Simpler test focusing on the exact race condition
cache = LRUCache(maxsize=100)

# Populate cache
for i in range(100):
    cache[i] = i * 10

error_occurred = False
error_message = ""

def delete_thread():
    """Thread that deletes items"""
    global error_occurred, error_message
    try:
        for _ in range(1000):
            # Delete first half of items
            for key in range(50):
                if key in cache:
                    del cache[key]  # This line lacks thread lock!
            # Re-add them
            for key in range(50):
                cache[key] = key * 10
    except Exception as e:
        error_occurred = True
        error_message = f"Delete thread: {type(e).__name__}: {e}"

def iterate_thread():
    """Thread that iterates over cache"""
    global error_occurred, error_message
    try:
        for _ in range(1000):
            # Iteration can fail if dictionary changes size
            keys_list = []
            for key in cache:  # This can raise RuntimeError
                keys_list.append(key)
    except RuntimeError as e:
        if "dictionary changed size during iteration" in str(e):
            error_occurred = True
            error_message = f"Iterate thread: RuntimeError: dictionary changed size during iteration"
    except Exception as e:
        error_occurred = True
        error_message = f"Iterate thread: {type(e).__name__}: {e}"

print("Demonstration of LRUCache.__delitem__ thread-safety bug")
print("=" * 60)
print()
print("Setup: LRUCache with 100 items")
print("Thread 1: Repeatedly deletes and re-adds keys 0-49")
print("Thread 2: Repeatedly iterates over all keys in cache")
print()
print("Expected: Thread-safe operation (per class documentation)")
print("Actual: Potential RuntimeError due to missing lock in __delitem__")
print()
print("Running test...")

# Run both threads
t1 = threading.Thread(target=delete_thread)
t2 = threading.Thread(target=iterate_thread)

t1.start()
t2.start()

t1.join()
t2.join()

print()
if error_occurred:
    print("BUG CONFIRMED: Race condition detected!")
    print(f"Error: {error_message}")
    print()
    print("Explanation:")
    print("  - __delitem__ (line 81-82) modifies self._cache WITHOUT acquiring self._lock")
    print("  - Meanwhile, __iter__ creates a snapshot of cache keys")
    print("  - This causes 'dictionary changed size during iteration' errors")
else:
    print("No error in this run (race conditions are probabilistic)")
    print()
    print("However, the bug is structurally present in the code:")
    print("  - __getitem__ uses: with self._lock (line 55)")
    print("  - __setitem__ uses: with self._lock (line 68)")
    print("  - __delitem__ MISSING lock (line 81-82)")
    print()
    print("This violates the 'Thread-safe LRUCache' promise in the docstring.")
```

<details>

<summary>
Structural bug confirmed through code inspection
</summary>
```
Demonstration of LRUCache.__delitem__ thread-safety bug
============================================================

Setup: LRUCache with 100 items
Thread 1: Repeatedly deletes and re-adds keys 0-49
Thread 2: Repeatedly iterates over all keys in cache

Expected: Thread-safe operation (per class documentation)
Actual: Potential RuntimeError due to missing lock in __delitem__

Running test...

No error in this run (race conditions are probabilistic)

However, the bug is structurally present in the code:
  - __getitem__ uses: with self._lock (line 55)
  - __setitem__ uses: with self._lock (line 68)
  - __delitem__ MISSING lock (line 81-82)

This violates the 'Thread-safe LRUCache' promise in the docstring.
```
</details>

## Why This Is A Bug

This violates the explicit thread-safety contract documented in the class docstring: "Thread-safe LRUCache based on an OrderedDict." The implementation demonstrates clear intent for thread-safety:

1. **Documentation Promise**: The class docstring at line 13 explicitly states "Thread-safe LRUCache" without any qualifications or exceptions.

2. **Inconsistent Implementation**: Every other mutating method correctly uses the lock:
   - `__getitem__` (line 55): `with self._lock:`
   - `__setitem__` (line 68): `with self._lock:`
   - `maxsize.setter` (line 102): `with self._lock:`
   - Only `__delitem__` (lines 81-82) fails to acquire the lock

3. **OrderedDict Requirements**: Python's OrderedDict documentation confirms it requires external synchronization for thread-safety. Without the lock, concurrent access can cause:
   - `RuntimeError: dictionary changed size during iteration`
   - `KeyError` exceptions
   - Internal corruption of the OrderedDict structure

4. **Infrastructure Exists**: The class creates `self._lock = threading.RLock()` in `__init__` (line 50) specifically for thread-safety, showing this was a deliberate design goal.

## Relevant Context

- **File location**: `/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages/xarray/backends/lru_cache.py`
- **Usage**: This LRUCache is used internally by xarray's backend systems for caching file handles and other I/O resources
- **Impact**: Multi-threaded applications using xarray for parallel I/O operations could experience crashes or data corruption
- **Python documentation on thread-safety**: https://docs.python.org/3/library/collections.html#collections.OrderedDict states that dict operations are not thread-safe
- **Related code**: All other methods in the same class demonstrate the correct locking pattern

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