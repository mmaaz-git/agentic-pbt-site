# Bug Report: xarray.backends CombinedLock.locked() Always Returns True

**Target**: `xarray.backends.locks.CombinedLock.locked()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `CombinedLock.locked()` method incorrectly returns `True` whenever the lock contains at least one constituent lock, regardless of whether any locks are actually acquired. This happens because line 236 references the `locked` method instead of calling it.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import threading
from xarray.backends.locks import CombinedLock

@given(st.lists(st.booleans(), min_size=1, max_size=5))
def test_combined_lock_locked_reflects_actual_state(lock_states):
    locks = [threading.Lock() for _ in lock_states]

    for lock, should_lock in zip(locks, lock_states):
        if should_lock:
            lock.acquire()

    combined = CombinedLock(locks)
    result = combined.locked()

    for lock, should_lock in zip(locks, lock_states):
        if should_lock:
            lock.release()

    assert isinstance(result, bool), f"locked() returned {type(result)} instead of bool"
    assert result == any(lock_states), f"locked() returned {result} but expected {any(lock_states)}"
```

**Failing input**: `lock_states = [False]` (or any list where all values are False)

## Reproducing the Bug

```python
import threading
from xarray.backends.locks import CombinedLock

lock1 = threading.Lock()
lock2 = threading.Lock()

combined = CombinedLock([lock1, lock2])

print(f"lock1.locked() = {lock1.locked()}")
print(f"lock2.locked() = {lock2.locked()}")
print(f"combined.locked() = {combined.locked()}")

assert combined.locked() == False
```

## Why This Is A Bug

The docstring for `CombinedLock` states: "Like a locked door, a CombinedLock is locked if any of its constituent locks are locked."

When no constituent locks are acquired, `combined.locked()` should return `False`. Instead, it returns `True` whenever the CombinedLock has at least one constituent lock.

This happens because line 236 uses `lock.locked` (a reference to the method object) instead of `lock.locked()` (calling the method). Method objects are always truthy in Python, so `any(lock.locked for lock in self.locks)` evaluates to `True` whenever `self.locks` is non-empty.

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