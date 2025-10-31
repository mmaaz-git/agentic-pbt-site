# Bug Report: xarray.backends.locks.CombinedLock.locked() Method Call Error

**Target**: `xarray.backends.locks.CombinedLock.locked()`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `CombinedLock.locked()` method incorrectly accesses `lock.locked` as an attribute instead of calling `lock.locked()` as a method, causing it to always return a truthy value when locks are present, regardless of their actual locked state.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

import threading
from hypothesis import given, strategies as st
from xarray.backends.locks import CombinedLock

@given(st.integers(min_value=1, max_value=10))
def test_combinedlock_locked_when_all_unlocked(n_locks):
    locks = [threading.Lock() for _ in range(n_locks)]
    combined = CombinedLock(locks)

    assert not combined.locked(), \
        "CombinedLock should return False when all constituent locks are unlocked"

@given(st.integers(min_value=1, max_value=10))
def test_combinedlock_locked_when_one_locked(n_locks):
    locks = [threading.Lock() for _ in range(n_locks)]
    combined = CombinedLock(locks)

    locks[0].acquire()
    try:
        assert combined.locked(), \
            "CombinedLock should return True when any constituent lock is locked"
    finally:
        locks[0].release()
```

**Failing input**: Any `n_locks >= 1` for the first test

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

import threading
from xarray.backends.locks import CombinedLock

locks = [threading.Lock(), threading.Lock()]
combined = CombinedLock(locks)

print("All locks unlocked")
print(f"combined.locked() = {combined.locked()}")

result = combined.locked()
print(f"Expected: False (or a boolean)")
print(f"Actual type: {type(result)}")
print(f"Actual value (as bool): {bool(result)}")
```

## Why This Is A Bug

The docstring for `CombinedLock` states (lines 211-214 in locks.py):
> "A combination of multiple locks. Like a locked door, a CombinedLock is locked if any of its constituent locks are locked."

The implementation in line 236 is:
```python
def locked(self):
    return any(lock.locked for lock in self.locks)
```

This accesses `lock.locked` as an attribute, which returns the bound method object, not the result of calling the method. Since bound method objects are always truthy (they're not None, 0, False, or empty), `any(lock.locked for lock in self.locks)` returns a truthy value whenever there are any locks present, regardless of whether they are actually locked.

The correct implementation should call the `locked()` method:
```python
def locked(self):
    return any(lock.locked() for lock in self.locks)
```

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