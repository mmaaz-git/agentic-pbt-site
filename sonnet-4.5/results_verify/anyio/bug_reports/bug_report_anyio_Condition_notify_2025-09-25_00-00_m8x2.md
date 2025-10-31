# Bug Report: anyio.abc.Condition.notify Documentation Contract Violation

**Target**: `anyio.abc.Condition.notify`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `Condition.notify(n)` method's docstring claims to "Notify exactly n listeners", but the implementation only notifies "at most n listeners" when there are fewer than n waiters.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import anyio
from anyio._core._synchronization import Condition


@given(st.integers(min_value=1, max_value=10))
def test_notify_exactly_n_property(n):
    condition = Condition()

    num_waiters = n - 1
    for _ in range(num_waiters):
        condition._waiters.append(anyio.Event())

    notified_count = 0
    for event in list(condition._waiters):
        if event.is_set():
            notified_count += 1

    condition.notify(n)

    actual_notified = sum(1 for e in list(condition._waiters)[:num_waiters] if e.is_set())
    assert actual_notified == n, f"Expected exactly {n} notifications, got {actual_notified}"
```

**Failing input**: Any `n` greater than the number of waiting tasks, e.g., `n=5` with only 3 waiters.

## Reproducing the Bug

```python
from anyio._core._synchronization import Condition
from anyio import Event

condition = Condition()

for _ in range(3):
    condition._waiters.append(Event())

print(f"Number of waiters before notify: {len(condition._waiters)}")

condition.notify(5)

notified = sum(1 for e in condition._waiters if e.is_set())
print(f"Number actually notified: {notified}")
print(f"Expected (per docstring 'exactly n'): 5")
print(f"Actual behavior: notified {notified} (at most n)")
```

**Output**:
```
Number of waiters before notify: 3
Number actually notified: 3
Expected (per docstring 'exactly n'): 5
Actual behavior: notified 3 (at most n)
```

## Why This Is A Bug

The docstring at `/lib/python3.13/site-packages/anyio/_core/_synchronization.py:306` states:

```python
def notify(self, n: int = 1) -> None:
    """Notify exactly n listeners."""
```

However, the implementation (lines 308-314) breaks early when there are fewer than n waiters:

```python
for _ in range(n):
    try:
        event = self._waiters.popleft()
    except IndexError:
        break  # Stops early if fewer than n waiters exist

    event.set()
```

The word "exactly" in the docstring creates a strong contract that the method will always notify precisely n listeners. When there are fewer than n waiters, the implementation notifies fewer listeners, violating this contract.

This contrasts with Python's standard library `threading.Condition.notify(n)` which correctly documents the behavior as "Wake up **at most** n threads".

## Fix

```diff
--- a/anyio/_core/_synchronization.py
+++ b/anyio/_core/_synchronization.py
@@ -303,7 +303,7 @@ class Condition:
         return self._lock.locked()

     def notify(self, n: int = 1) -> None:
-        """Notify exactly n listeners."""
+        """Notify at most n listeners."""
         self._check_acquired()
         for _ in range(n):
             try:
```