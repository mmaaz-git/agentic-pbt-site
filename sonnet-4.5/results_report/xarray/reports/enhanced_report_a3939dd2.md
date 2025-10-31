# Bug Report: xarray.backends.locks.CombinedLock.locked() Always Returns True

**Target**: `xarray.backends.locks.CombinedLock.locked()`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `CombinedLock.locked()` method incorrectly accesses `lock.locked` as a property instead of calling `lock.locked()` as a method, causing it to always return `True` when any locks are present, regardless of whether they are actually acquired.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from xarray.backends.locks import CombinedLock
import threading

@given(num_locks=st.integers(min_value=1, max_value=5))
def test_combined_lock_locked_returns_correct_state(num_locks):
    """Test that CombinedLock.locked() correctly returns False when no locks are acquired."""
    locks = [threading.Lock() for _ in range(num_locks)]
    combined = CombinedLock(locks)

    # When no locks are acquired, locked() should return False
    assert combined.locked() == False, f"CombinedLock.locked() returned True when no locks are acquired (expected False)"

    # Acquire the first lock
    locks[0].acquire()
    try:
        # When at least one lock is acquired, locked() should return True
        assert combined.locked() == True, f"CombinedLock.locked() returned False when a lock is acquired (expected True)"
    finally:
        locks[0].release()

    # After releasing, locked() should return False again
    assert combined.locked() == False, f"CombinedLock.locked() returned True after releasing all locks (expected False)"

if __name__ == "__main__":
    # Run the test with hypothesis
    test_combined_lock_locked_returns_correct_state()
```

<details>

<summary>
**Failing input**: `num_locks=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 27, in <module>
    test_combined_lock_locked_returns_correct_state()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 6, in test_combined_lock_locked_returns_correct_state
    def test_combined_lock_locked_returns_correct_state(num_locks):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 12, in test_combined_lock_locked_returns_correct_state
    assert combined.locked() == False, f"CombinedLock.locked() returned True when no locks are acquired (expected False)"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: CombinedLock.locked() returned True when no locks are acquired (expected False)
Falsifying example: test_combined_lock_locked_returns_correct_state(
    num_locks=1,
)
```
</details>

## Reproducing the Bug

```python
import threading
from xarray.backends.locks import CombinedLock

# Create two threading locks
lock1 = threading.Lock()
lock2 = threading.Lock()

# Create a CombinedLock with these locks
combined = CombinedLock([lock1, lock2])

# Test 1: Check if CombinedLock reports locked when no locks are acquired
print("Test 1: No locks acquired")
print(f"combined.locked() returns: {combined.locked()}")
print(f"Expected: False")
print()

# Test 2: Acquire one lock and check
lock1.acquire()
print("Test 2: lock1 acquired")
print(f"combined.locked() returns: {combined.locked()}")
print(f"Expected: True")
lock1.release()
print()

# Test 3: Show the actual bug - accessing lock.locked without parentheses
print("Test 3: Demonstrating the bug")
print(f"lock1.locked (without parentheses): {lock1.locked}")
print(f"lock1.locked() (with parentheses): {lock1.locked()}")
print()

# Test 4: Show what the bug actually evaluates
print("Test 4: What the buggy code evaluates")
print(f"any([lock1.locked, lock2.locked]): {any([lock1.locked, lock2.locked])}")
print(f"any([lock1.locked(), lock2.locked()]): {any([lock1.locked(), lock2.locked()])}")
```

<details>

<summary>
CombinedLock.locked() incorrectly returns True when no locks are acquired
</summary>
```
Test 1: No locks acquired
combined.locked() returns: True
Expected: False

Test 2: lock1 acquired
combined.locked() returns: True
Expected: True

Test 3: Demonstrating the bug
lock1.locked (without parentheses): <built-in method locked of _thread.lock object at 0x7b5be44ad890>
lock1.locked() (with parentheses): False

Test 4: What the buggy code evaluates
any([lock1.locked, lock2.locked]): True
any([lock1.locked(), lock2.locked()]): False
```
</details>

## Why This Is A Bug

The bug violates the documented behavior of `CombinedLock` which states in its docstring: "Like a locked door, a CombinedLock is locked if any of its constituent locks are locked." The implementation on line 236 of `xarray/backends/locks.py` uses `lock.locked` instead of `lock.locked()`, which has several critical consequences:

1. **`threading.Lock.locked` is a method, not a property**: The Python standard library's `threading.Lock` implements `locked()` as a method that must be called with parentheses to return a boolean value.

2. **Method objects are always truthy**: When accessed without parentheses, `lock.locked` returns a bound method object like `<built-in method locked of _thread.lock object>`. These method objects always evaluate to `True` in boolean contexts.

3. **`any()` evaluates the wrong thing**: The expression `any(lock.locked for lock in self.locks)` checks if any method object exists (always True when locks exist), not whether any lock is actually acquired.

4. **Inconsistent with other implementations**: The same file correctly implements `SerializableLock.locked()` on line 70 with `return self.lock.locked()`, using proper method call syntax.

This causes `CombinedLock.locked()` to return incorrect results:
- Returns `True` when no locks are acquired (should be `False`)
- Returns `True` when all locks are released (should be `False`)
- Only returns `False` when the CombinedLock has no locks at all

## Relevant Context

This bug affects thread synchronization in xarray's file I/O operations. The xarray library uses `CombinedLock` to manage concurrent access to HDF5 and NetCDF files, which are not thread-safe. The `HDF5_LOCK` and `NETCDFC_LOCK` defined in this module rely on proper lock behavior to prevent data corruption.

The bug has likely gone unnoticed because:
1. The acquire/release methods work correctly, so basic locking still functions
2. Code may not frequently check the locked state directly
3. The method name and interface match expectations, making the bug subtle

Related code locations:
- Correct implementation: `xarray/backends/locks.py:70` in `SerializableLock.locked()`
- Bug location: `xarray/backends/locks.py:236` in `CombinedLock.locked()`
- Usage context: HDF5 and NetCDF file locking throughout xarray's backend modules

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