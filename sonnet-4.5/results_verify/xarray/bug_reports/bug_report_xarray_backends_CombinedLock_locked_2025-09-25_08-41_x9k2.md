# Bug Report: xarray.backends.locks.CombinedLock.locked() Method Call Error

**Target**: `xarray.backends.locks.CombinedLock.locked()`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `CombinedLock.locked()` method attempts to access `lock.locked` as a property instead of calling the `locked()` method, causing an AttributeError when used with `threading.Lock` objects.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from xarray.backends.locks import CombinedLock
import threading

@given(num_locks=st.integers(min_value=1, max_value=5))
def test_combined_lock_locked_property(num_locks):
    locks = [threading.Lock() for _ in range(num_locks)]
    combined = CombinedLock(locks)

    result = combined.locked()
    assert isinstance(result, bool), "locked() should return a boolean"
```

**Failing input**: Any `CombinedLock` created with `threading.Lock` objects

## Reproducing the Bug

```python
import threading
from xarray.backends.locks import CombinedLock

lock1 = threading.Lock()
lock2 = threading.Lock()
combined = CombinedLock([lock1, lock2])

lock1.acquire()
try:
    result = combined.locked()
except AttributeError as e:
    print(f"AttributeError: {e}")
finally:
    lock1.release()
```

## Why This Is A Bug

The `CombinedLock.locked()` method on line 236 of `xarray/backends/locks.py` uses `lock.locked` (property access) instead of `lock.locked()` (method call). The `threading.Lock.locked()` is a method, not a property, so accessing it without parentheses returns the bound method object rather than calling it. This causes `any()` to evaluate the method objects themselves (which are truthy) instead of their return values.

This is confirmed by comparing with the correct usage in `SerializableLock.locked()` on line 70 of the same file, which correctly calls `self.lock.locked()`.

## Fix

```diff
--- a/xarray/backends/locks.py
+++ b/xarray/backends/locks.py
@@ -233,7 +233,7 @@ class CombinedLock:
             lock.__exit__(*args)

     def locked(self):
-        return any(lock.locked for lock in self.locks)
+        return any(lock.locked() for lock in self.locks)

     def __repr__(self):
         return f"CombinedLock({list(self.locks)!r})"
```