# Bug Report: xarray.backends.locks.CombinedLock.locked() Always Returns True

**Target**: `xarray.backends.locks.CombinedLock.locked`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `CombinedLock.locked()` method always returns `True` regardless of whether any constituent locks are actually locked, due to accessing the method object instead of calling it.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
import threading
from xarray.backends.locks import CombinedLock

@settings(max_examples=200)
@given(locks_count=st.integers(min_value=0, max_value=10))
def test_combined_lock_locked_property(locks_count):
    locks = [threading.Lock() for _ in range(locks_count)]
    combined = CombinedLock(locks)

    assert not combined.locked()

    if locks:
        locks[0].acquire()
        assert combined.locked()
        locks[0].release()
```

**Failing input**: `locks_count=1`

## Reproducing the Bug

```python
import threading
from xarray.backends.locks import CombinedLock

lock = threading.Lock()
combined = CombinedLock([lock])

print(f"Individual lock is locked: {lock.locked()}")
print(f"CombinedLock.locked(): {combined.locked()}")
```

Output:
```
Individual lock is locked: False
CombinedLock.locked(): True
```

## Why This Is A Bug

The method should return `True` only if at least one of the constituent locks is actually locked. Instead, it always returns `True` because it's checking the truthiness of the method object `lock.locked` (which is always truthy) rather than calling the method `lock.locked()`.

This breaks the lock API contract and could lead to incorrect synchronization behavior where code thinks a lock is held when it isn't, potentially causing race conditions.

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
