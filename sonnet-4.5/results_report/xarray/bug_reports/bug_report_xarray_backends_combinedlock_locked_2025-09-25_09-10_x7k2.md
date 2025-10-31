# Bug Report: xarray.backends.locks.CombinedLock.locked Always Returns True

**Target**: `xarray.backends.locks.CombinedLock.locked`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `CombinedLock.locked()` method always returns `True` when the lock has at least one constituent lock, even when none of the locks are actually locked. This is because it checks the method reference `lock.locked` instead of calling the method `lock.locked()`.

## Property-Based Test

```python
import threading
from hypothesis import given, strategies as st, settings
from xarray.backends.locks import CombinedLock

@given(st.lists(st.booleans(), min_size=1, max_size=10))
@settings(max_examples=200)
def test_combined_lock_locked_state(lock_states):
    locks = []
    for locked in lock_states:
        lock = threading.Lock()
        if locked:
            lock.acquire()
        locks.append(lock)

    combined = CombinedLock(locks)

    expected_locked = any(lock_states)
    actual_locked = combined.locked()

    for i, lock in enumerate(locks):
        if lock_states[i]:
            lock.release()

    assert actual_locked == expected_locked
```

**Failing input**: `lock_states=[False]`

## Reproducing the Bug

```python
import threading
from xarray.backends.locks import CombinedLock

lock = threading.Lock()
combined = CombinedLock([lock])

print(f"lock.locked() = {lock.locked()}")
print(f"combined.locked() = {combined.locked()}")
```

Output:
```
lock.locked() = False
combined.locked() = True
```

The `CombinedLock` reports being locked even though its constituent lock is not locked.

## Why This Is A Bug

According to the class docstring at line 211-214:

> Like a locked door, a CombinedLock is locked if any of its constituent locks are locked.

The current implementation violates this contract. When no constituent locks are locked, `CombinedLock.locked()` should return `False`, but it returns `True` instead.

This happens because line 236 uses:
```python
return any(lock.locked for lock in self.locks)
```

In Python, `lock.locked` is a bound method object, which is always truthy. The code should call the method with `lock.locked()`.

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