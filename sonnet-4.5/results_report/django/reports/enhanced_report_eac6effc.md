# Bug Report: django.core.mail.backends.console Returns None for Empty Messages

**Target**: `django.core.mail.backends.console.EmailBackend.send_messages`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `send_messages()` method in Django's ConsoleBackend returns `None` instead of `0` when called with an empty list of messages, violating the documented API contract that requires returning an integer count of messages sent.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Property-based test using Hypothesis to verify that Django's ConsoleBackend
always returns an integer from send_messages(), even with empty lists.
"""

import sys
import io
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from django.core.mail.backends.console import EmailBackend as ConsoleBackend

@given(st.integers(min_value=0, max_value=10))
def test_console_backend_empty_list_returns_int(n):
    """Test that ConsoleBackend.send_messages([]) returns an integer."""
    backend = ConsoleBackend(stream=io.StringIO())
    result = backend.send_messages([])
    assert isinstance(result, int), f"Expected int, got {type(result).__name__}: {result}"
    assert result == 0, f"Expected 0 for empty list, got {result}"

# Run the test
if __name__ == "__main__":
    test_console_backend_empty_list_returns_int()
```

<details>

<summary>
**Failing input**: `n=0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/47/hypo.py", line 24, in <module>
    test_console_backend_empty_list_returns_int()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/47/hypo.py", line 15, in test_console_backend_empty_list_returns_int
    def test_console_backend_empty_list_returns_int(n):
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/47/hypo.py", line 19, in test_console_backend_empty_list_returns_int
    assert isinstance(result, int), f"Expected int, got {type(result).__name__}: {result}"
           ~~~~~~~~~~^^^^^^^^^^^^^
AssertionError: Expected int, got NoneType: None
Falsifying example: test_console_backend_empty_list_returns_int(
    n=0,
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Minimal reproduction of Django console email backend bug.
This demonstrates that ConsoleBackend.send_messages() returns None
instead of 0 when called with an empty list of messages.
"""

import io
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.core.mail.backends.console import EmailBackend

# Create backend with StringIO to capture output
backend = EmailBackend(stream=io.StringIO())

# Call send_messages with empty list
result = backend.send_messages([])

# Print results
print(f"Result: {result}")
print(f"Type: {type(result).__name__}")

# Verify the bug
try:
    assert isinstance(result, int), f"Expected int, got {type(result)}"
    assert result == 0, f"Expected 0, got {result}"
    print("✓ Test passed")
except AssertionError as e:
    print(f"✗ Assertion failed: {e}")

# Also test with other backends for comparison
print("\n--- Comparison with other backends ---")

from django.core.mail.backends.dummy import EmailBackend as DummyBackend
from django.core.mail.backends.locmem import EmailBackend as LocmemBackend

dummy_backend = DummyBackend()
dummy_result = dummy_backend.send_messages([])
print(f"Dummy backend: {dummy_result} (type: {type(dummy_result).__name__})")

locmem_backend = LocmemBackend()
locmem_result = locmem_backend.send_messages([])
print(f"Locmem backend: {locmem_result} (type: {type(locmem_result).__name__})")
```

<details>

<summary>
ConsoleBackend returns None instead of 0 for empty message list
</summary>
```
Result: None
Type: NoneType
✗ Assertion failed: Expected int, got <class 'NoneType'>

--- Comparison with other backends ---
Dummy backend: 0 (type: int)
Locmem backend: 0 (type: int)
```
</details>

## Why This Is A Bug

This violates the explicit API contract documented in Django's base email backend class. The `BaseEmailBackend.send_messages()` method at `django/core/mail/backends/base.py:55-59` clearly states:

```python
def send_messages(self, email_messages):
    """
    Send one or more EmailMessage objects and return the number of email
    messages sent.
    """
```

The documentation requires returning "the number of email messages sent" - which must be an integer. When no messages are sent from an empty list, that number should be `0`, not `None`. This creates an inconsistency across Django's email backends:

- **SMTP Backend**: Returns `0` for empty lists (line 126: `return 0`)
- **Dummy Backend**: Returns `0` via `len(list(email_messages))`
- **Locmem Backend**: Returns `0` (initializes `msg_count = 0` and returns it)
- **Console Backend**: Returns `None` (line 31: bare `return` statement)
- **File-based Backend**: Inherits from Console, also returns `None`

This type inconsistency can cause subtle bugs in applications that switch between backends or rely on the documented integer return type. Code that works correctly with SMTP, Dummy, or Locmem backends may fail when using Console or File-based backends due to this contract violation.

## Relevant Context

The bug exists in `django/core/mail/backends/console.py` at lines 30-31:

```python
def send_messages(self, email_messages):
    """Write all messages to the stream in a thread-safe way."""
    if not email_messages:
        return  # This returns None, should return 0
    msg_count = 0
    # ... rest of the method
```

The bare `return` statement in Python implicitly returns `None`. This is inconsistent with the rest of the method which correctly returns `msg_count` (an integer) at line 45. The File-based backend inherits this bug since it extends ConsoleBackend.

Documentation link: https://docs.djangoproject.com/en/stable/topics/email/#email-backends

## Proposed Fix

```diff
--- a/django/core/mail/backends/console.py
+++ b/django/core/mail/backends/console.py
@@ -28,7 +28,7 @@ class EmailBackend(BaseEmailBackend):
     def send_messages(self, email_messages):
         """Write all messages to the stream in a thread-safe way."""
         if not email_messages:
-            return
+            return 0
         msg_count = 0
         with self._lock:
             try:
```