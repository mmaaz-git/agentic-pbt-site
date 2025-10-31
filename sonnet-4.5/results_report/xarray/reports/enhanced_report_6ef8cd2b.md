# Bug Report: xarray.backends.locks.CombinedLock.locked Always Returns True When Locks Present

**Target**: `xarray.backends.locks.CombinedLock.locked`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `CombinedLock.locked()` method incorrectly returns `True` whenever it has constituent locks, regardless of whether any locks are actually acquired, because it checks the method reference `lock.locked` instead of calling `lock.locked()`.

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

# Run the test
test_combined_lock_locked_state()
```

<details>

<summary>
**Failing input**: `lock_states=[False]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/63/hypo.py", line 27, in <module>
    test_combined_lock_locked_state()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/63/hypo.py", line 6, in test_combined_lock_locked_state
    @settings(max_examples=200)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/63/hypo.py", line 24, in test_combined_lock_locked_state
    assert actual_locked == expected_locked
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_combined_lock_locked_state(
    lock_states=[False],
)
```
</details>

## Reproducing the Bug

```python
import threading
from xarray.backends.locks import CombinedLock

# Create a simple threading lock
lock = threading.Lock()

# Create a CombinedLock with the single lock
combined = CombinedLock([lock])

# Check if the individual lock is locked (should be False since we haven't acquired it)
print(f"lock.locked() = {lock.locked()}")

# Check if the CombinedLock is locked (should also be False, but will be True due to bug)
print(f"combined.locked() = {combined.locked()}")

# Now let's test with the lock actually acquired
print("\n--- After acquiring the lock ---")
lock.acquire()
print(f"lock.locked() = {lock.locked()}")
print(f"combined.locked() = {combined.locked()}")
lock.release()

# Test with multiple locks, none acquired
print("\n--- Multiple locks, none acquired ---")
lock1 = threading.Lock()
lock2 = threading.Lock()
lock3 = threading.Lock()
combined_multi = CombinedLock([lock1, lock2, lock3])
print(f"lock1.locked() = {lock1.locked()}")
print(f"lock2.locked() = {lock2.locked()}")
print(f"lock3.locked() = {lock3.locked()}")
print(f"combined_multi.locked() = {combined_multi.locked()}")

# Test with one lock acquired
print("\n--- Multiple locks, one acquired ---")
lock2.acquire()
print(f"lock1.locked() = {lock1.locked()}")
print(f"lock2.locked() = {lock2.locked()}")
print(f"lock3.locked() = {lock3.locked()}")
print(f"combined_multi.locked() = {combined_multi.locked()}")
lock2.release()
```

<details>

<summary>
CombinedLock reports locked=True even when all constituent locks are unlocked
</summary>
```
lock.locked() = False
combined.locked() = True

--- After acquiring the lock ---
lock.locked() = True
combined.locked() = True

--- Multiple locks, none acquired ---
lock1.locked() = False
lock2.locked() = False
lock3.locked() = False
combined_multi.locked() = True

--- Multiple locks, one acquired ---
lock1.locked() = False
lock2.locked() = True
lock3.locked() = False
combined_multi.locked() = True
```
</details>

## Why This Is A Bug

This violates the documented behavior stated in the CombinedLock class docstring (lines 211-214 of `/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages/xarray/backends/locks.py`):

> "Like a locked door, a CombinedLock is locked if any of its constituent locks are locked."

The implementation on line 236 checks `lock.locked` which evaluates the truthiness of the bound method object (always `True` for existing methods), rather than calling `lock.locked()` to get the actual lock state. This means:

1. When NO constituent locks are acquired, `combined.locked()` incorrectly returns `True`
2. The method cannot distinguish between "no locks acquired" and "some locks acquired"
3. Code relying on this method to check resource availability gets incorrect information
4. This breaks the standard Python locking protocol where `locked()` should accurately report lock state

## Relevant Context

The bug occurs in a utility class used by xarray's backend systems for managing concurrent access to data files. The `CombinedLock` class is part of the public API (`xarray.backends.locks.CombinedLock`) and implements the standard Python context manager and locking protocols.

The class is designed to aggregate multiple locks (e.g., for HDF5 and NetCDF-C libraries which are not thread-safe) and treat them as a single lock. The `locked()` method is crucial for checking resource availability in concurrent/distributed computing scenarios.

Key code location: `/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages/xarray/backends/locks.py:236`

Related classes in the same module:
- `SerializableLock` (lines 15-81): Correctly implements `locked()` by calling `self.lock.locked()` on line 70
- `DummyLock` (lines 242-258): Correctly implements `locked()` by returning `False` on line 258

## Proposed Fix

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