# Bug Report: django.core.mail.backends.console Returns None Instead of 0 for Empty Message Lists

**Target**: `django.core.mail.backends.console.EmailBackend.send_messages`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `send_messages` method in Django's console email backend returns `None` instead of `0` when called with an empty list of messages, violating the documented API contract that requires returning "the number of email messages sent."

## Property-Based Test

```python
#!/usr/bin/env python3
"""Property-based test that discovers the console EmailBackend bug."""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

import io
from hypothesis import given, strategies as st
from django.core.mail.backends.console import EmailBackend as ConsoleBackend
from unittest.mock import Mock

@given(st.lists(st.just(Mock(message=lambda: Mock(as_bytes=lambda: b'test', get_charset=lambda: None))), max_size=10))
def test_console_backend_return_type_invariant(messages):
    stream = io.StringIO()
    backend = ConsoleBackend(stream=stream, fail_silently=False)
    result = backend.send_messages(messages)

    assert result is not None, f"send_messages should return an integer, not None"
    assert isinstance(result, int), f"send_messages should return an integer"
    assert result == len(messages), f"send_messages should return message count"

# Run the test
if __name__ == "__main__":
    test_console_backend_return_type_invariant()
```

<details>

<summary>
**Failing input**: `[]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/15/hypo.py", line 24, in <module>
    test_console_backend_return_type_invariant()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/15/hypo.py", line 13, in test_console_backend_return_type_invariant
    def test_console_backend_return_type_invariant(messages):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/15/hypo.py", line 18, in test_console_backend_return_type_invariant
    assert result is not None, f"send_messages should return an integer, not None"
           ^^^^^^^^^^^^^^^^^^
AssertionError: send_messages should return an integer, not None
Falsifying example: test_console_backend_return_type_invariant(
    messages=[],
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal demonstration of the console EmailBackend bug."""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

import io
from django.core.mail.backends.console import EmailBackend

# Create a console backend with a string stream
stream = io.StringIO()
backend = EmailBackend(stream=stream)

# Call send_messages with an empty list
result = backend.send_messages([])

# Show the result
print(f"Result from send_messages([]): {result!r}")
print(f"Expected result: 0")
print(f"Type of result: {type(result)}")

# Test that it equals 0 (this will fail)
try:
    assert result == 0, f"Expected 0 but got {result!r}"
    print("PASS: Assertion succeeded")
except AssertionError as e:
    print(f"FAIL: {e}")

# Also demonstrate the issue with arithmetic operations
try:
    total = 0
    total += result  # This will fail with TypeError if result is None
    print(f"Arithmetic test passed: 0 + result = {total}")
except TypeError as e:
    print(f"FAIL: Arithmetic operation failed with TypeError: {e}")
```

<details>

<summary>
Returns None instead of 0, causing TypeError in arithmetic operations
</summary>
```
Result from send_messages([]): None
Expected result: 0
Type of result: <class 'NoneType'>
FAIL: Expected 0 but got None
FAIL: Arithmetic operation failed with TypeError: unsupported operand type(s) for +=: 'int' and 'NoneType'
```
</details>

## Why This Is A Bug

This violates Django's documented email backend API contract in multiple ways:

1. **Explicit Documentation Violation**: The `BaseEmailBackend.send_messages()` docstring at `/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/core/mail/backends/base.py:57-58` explicitly states: "Send one or more EmailMessage objects and **return the number of email messages sent**." The number of messages sent from an empty list is 0, not None.

2. **Inconsistent Backend Behavior**: The SMTP backend correctly implements the contract at `/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/core/mail/backends/smtp.py:125-126` with `if not email_messages: return 0`. The console backend at `/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/core/mail/backends/console.py:30-31` incorrectly uses a bare `return` statement, returning None.

3. **Type Error Risk**: Code that performs arithmetic on the return value (e.g., `total_sent += backend.send_messages(batch)`) will raise `TypeError: unsupported operand type(s) for +=: 'int' and 'NoneType'` when the batch is empty.

4. **Backend Interchangeability Broken**: Django's email backends are designed to be pluggable and interchangeable. Having one backend return None while others return 0 for the same input breaks this design principle.

5. **Inheritance Propagation**: The `filebased.EmailBackend` inherits from `console.EmailBackend` without overriding `send_messages()`, so it inherits this bug, affecting multiple backends in Django's email system.

## Relevant Context

The bug is located at line 31 of `/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/core/mail/backends/console.py`:

```python
def send_messages(self, email_messages):
    """Write all messages to the stream in a thread-safe way."""
    if not email_messages:
        return  # <-- This returns None, should return 0
    msg_count = 0
    # ... rest of method that correctly returns msg_count
```

This affects any code using Django's console or file-based email backends in scenarios such as:
- Batch email processing with potentially empty batches
- Testing email functionality with edge cases
- Conditional email sending that may result in empty message lists
- Email statistics tracking that sums sent counts

Documentation references:
- Base class contract: `django/core/mail/backends/base.py:55-62`
- SMTP correct implementation: `django/core/mail/backends/smtp.py:120-126`
- Console buggy implementation: `django/core/mail/backends/console.py:28-45`
- File backend inheritance: `django/core/mail/backends/filebased.py:11`

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