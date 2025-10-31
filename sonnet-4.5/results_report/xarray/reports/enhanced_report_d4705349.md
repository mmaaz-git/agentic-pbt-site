# Bug Report: xarray.backends.locks CombinedLock.locked() Always Returns True for Non-Empty Lock Lists

**Target**: `xarray.backends.locks.CombinedLock.locked()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `CombinedLock.locked()` method incorrectly returns `True` whenever it contains at least one constituent lock, regardless of whether any locks are actually acquired. This is caused by a missing function call on line 236 where `lock.locked` (method reference) is used instead of `lock.locked()` (method call).

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

if __name__ == "__main__":
    test_combined_lock_locked_reflects_actual_state()
```

<details>

<summary>
**Failing input**: `lock_states=[False]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/42/hypo.py", line 24, in <module>
    test_combined_lock_locked_reflects_actual_state()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/42/hypo.py", line 6, in test_combined_lock_locked_reflects_actual_state
    def test_combined_lock_locked_reflects_actual_state(lock_states):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/42/hypo.py", line 21, in test_combined_lock_locked_reflects_actual_state
    assert result == any(lock_states), f"locked() returned {result} but expected {any(lock_states)}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: locked() returned True but expected False
Falsifying example: test_combined_lock_locked_reflects_actual_state(
    lock_states=[False],
)
```
</details>

## Reproducing the Bug

```python
import threading
from xarray.backends.locks import CombinedLock

# Create two regular threading locks
lock1 = threading.Lock()
lock2 = threading.Lock()

# Create a CombinedLock from them
combined = CombinedLock([lock1, lock2])

# Check the status of individual locks (should be False as they're not acquired)
print(f"lock1.locked() = {lock1.locked()}")
print(f"lock2.locked() = {lock2.locked()}")

# Check the combined lock status (should be False but returns True due to bug)
print(f"combined.locked() = {combined.locked()}")

# This assertion should pass but fails due to the bug
assert combined.locked() == False, f"Expected combined.locked() to be False, but got {combined.locked()}"
```

<details>

<summary>
AssertionError: Expected combined.locked() to be False
</summary>
```
lock1.locked() = False
lock2.locked() = False
combined.locked() = True
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/42/repo.py", line 19, in <module>
    assert combined.locked() == False, f"Expected combined.locked() to be False, but got {combined.locked()}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Expected combined.locked() to be False, but got True
```
</details>

## Why This Is A Bug

The `CombinedLock` class documentation explicitly states: "Like a locked door, a CombinedLock is locked if any of its constituent locks are locked." This clearly implies that when ALL constituent locks are unlocked, the combined lock should report as unlocked (`False`).

The bug occurs because line 236 in `/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages/xarray/backends/locks.py` uses `lock.locked` instead of `lock.locked()`. In Python, `lock.locked` is a method object which is always truthy when evaluated in a boolean context. Therefore, `any(lock.locked for lock in self.locks)` will return `True` whenever `self.locks` contains at least one lock, regardless of the actual lock states.

This violates the documented contract and basic expectations about lock state reporting. Code relying on `CombinedLock.locked()` to determine if any locks are held will receive incorrect information, potentially leading to synchronization bugs or debugging confusion.

## Relevant Context

The `CombinedLock` class is used internally by xarray for managing multiple locks in parallel I/O operations, particularly with dask. The class appears at line 210-239 in `xarray/backends/locks.py`.

Notably, there are no existing tests for the `CombinedLock.locked()` method in the xarray test suite, which explains how this bug went undetected. The `locked()` method is likely not heavily used in practice, as most lock usage follows the context manager pattern (`with lock:`) rather than checking lock state.

The bug is trivial to verify - in Python, any method reference evaluates to `True`:
- `lock.locked` → `<bound method Lock.locked of <unlocked _thread.lock object>>` (truthy)
- `lock.locked()` → `False` (the actual lock state)

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