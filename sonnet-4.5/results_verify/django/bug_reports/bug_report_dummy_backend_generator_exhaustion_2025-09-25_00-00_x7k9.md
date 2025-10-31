# Bug Report: Dummy Backend Generator Exhaustion

**Target**: `django.core.mail.backends.dummy.EmailBackend.send_messages`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The dummy email backend exhausts generator arguments by converting them to lists, violating its documented behavior of "doing nothing" and breaking the iterator protocol contract.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.core.mail.backends.dummy import EmailBackend

@given(st.lists(st.integers(), min_size=0, max_size=100))
def test_dummy_backend_does_not_exhaust_generators(message_count):
    backend = EmailBackend()

    exhausted = False
    def message_generator():
        nonlocal exhausted
        for i in range(message_count):
            yield object()
        exhausted = True

    gen = message_generator()
    result = backend.send_messages(gen)

    assert result == len(message_count)
    assert not exhausted, "Backend should not exhaust generators"
```

**Failing input**: Any generator (e.g., a generator of 5 messages)

## Reproducing the Bug

```python
from django.core.mail.backends.dummy import EmailBackend

backend = EmailBackend()

exhausted = False
def message_generator():
    global exhausted
    for i in range(3):
        yield object()
    exhausted = True

gen = message_generator()
result = backend.send_messages(gen)

print(f"Exhausted: {exhausted}")
assert exhausted == True, "Bug: Generator was exhausted!"
```

## Why This Is A Bug

1. **Contract violation**: The module docstring states "Dummy email backend that does nothing" (line 2), but converting an iterable to a list is not "doing nothing" - it consumes the entire iterator.

2. **Breaks iterator protocol**: Callers may pass generators that should not be exhausted. The backend claims to be a no-op but has observable side effects.

3. **Inconsistent with other backends**: Console, SMTP, and other backends iterate through messages without converting to list first, preserving the iterator.

4. **Unnecessary operation**: The backend could simply iterate and count without converting to list:
   ```python
   sum(1 for _ in email_messages)
   ```

## Fix

```diff
--- a/django/core/mail/backends/dummy.py
+++ b/django/core/mail/backends/dummy.py
@@ -7,4 +7,4 @@ from django.core.mail.backends.base import BaseEmailBackend

 class EmailBackend(BaseEmailBackend):
     def send_messages(self, email_messages):
-        return len(list(email_messages))
+        return sum(1 for _ in email_messages)
```