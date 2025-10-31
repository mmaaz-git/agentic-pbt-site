# Bug Report: xarray.backends.locks.CombinedLock.locked() Missing Method Call

**Target**: `xarray.backends.locks.CombinedLock.locked`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `CombinedLock.locked()` method fails to call the `locked()` method on constituent locks, instead checking the truthiness of the method object itself. This causes the method to always return `True` when locks exist, regardless of whether they are actually locked.

## Property-Based Test

```python
import threading
from hypothesis import given, strategies as st
from xarray.backends.locks import CombinedLock


@given(st.integers(min_value=1, max_value=10))
def test_combined_lock_locked_false_when_no_locks_held(num_locks):
    locks = [threading.Lock() for _ in range(num_locks)]
    combined = CombinedLock(locks)

    assert combined.locked() == False, (
        "CombinedLock.locked() should return False when no locks are held"
    )
```

**Failing input**: `num_locks=1` (or any positive integer)

## Reproducing the Bug

```python
import threading
from xarray.backends.locks import CombinedLock

lock1 = threading.Lock()
lock2 = threading.Lock()
combined = CombinedLock([lock1, lock2])

print(combined.locked())
```

Output: `True` (expected: `False`)

The bug occurs because line 236 in `locks.py` calls `lock.locked` without parentheses:

```python
return any(lock.locked for lock in self.locks)
```

This evaluates the truthiness of the method object (always `True`), not the lock's actual state.

## Why This Is A Bug

The docstring states: "Like a locked door, a CombinedLock is locked if any of its constituent locks are locked."

The current implementation violates this contract by returning `True` even when no constituent locks are actually held. All lock types used with `CombinedLock` (`threading.Lock`, `SerializableLock`, `multiprocessing.Lock`, `DummyLock`) have `locked` as a method that must be called, not a property.

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