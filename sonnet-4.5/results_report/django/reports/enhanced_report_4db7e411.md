# Bug Report: Django ServerHandler Negative Content-Length Silent Data Loss

**Target**: `django.core.servers.basehttp.ServerHandler.__init__`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Django's ServerHandler class silently accepts negative Content-Length values from HTTP headers, causing the LimitedStream to immediately return empty data for all read operations, resulting in complete loss of valid request body data.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

import django
from django.conf import settings
if not settings.configured:
    settings.configure(DEBUG=True, SECRET_KEY='test')
    django.setup()

from hypothesis import given, strategies as st
import io
from django.core.servers.basehttp import ServerHandler

@given(st.integers(max_value=-1))
def test_serverhandler_rejects_negative_content_length(negative_length):
    environ = {"CONTENT_LENGTH": str(negative_length)}
    stdin = io.BytesIO(b"valid request body data")
    stdout = io.BytesIO()
    stderr = io.BytesIO()

    handler = ServerHandler(stdin, stdout, stderr, environ)

    # Property: Content-length should never be negative
    assert handler.stdin.limit >= 0, \
        f"LimitedStream.limit should be >= 0, got {handler.stdin.limit}"

    # Property: Valid data should be readable when present
    data = handler.stdin.read(10)
    assert len(data) > 0, "Should be able to read data when stream has content"

if __name__ == "__main__":
    test_serverhandler_rejects_negative_content_length()
```

<details>

<summary>
**Failing input**: `negative_length=-1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/41/hypo.py", line 32, in <module>
    test_serverhandler_rejects_negative_content_length()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/41/hypo.py", line 15, in test_serverhandler_rejects_negative_content_length
    def test_serverhandler_rejects_negative_content_length(negative_length):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/41/hypo.py", line 24, in test_serverhandler_rejects_negative_content_length
    assert handler.stdin.limit >= 0, \
           ^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: LimitedStream.limit should be >= 0, got -1
Falsifying example: test_serverhandler_rejects_negative_content_length(
    negative_length=-1,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

import django
from django.conf import settings
if not settings.configured:
    settings.configure(DEBUG=True, SECRET_KEY='test')
    django.setup()

import io
from django.core.servers.basehttp import ServerHandler

# Test case with negative Content-Length
environ = {"CONTENT_LENGTH": "-100"}
stdin = io.BytesIO(b"valid request body data")
stdout = io.BytesIO()
stderr = io.BytesIO()

handler = ServerHandler(stdin, stdout, stderr, environ)

print(f"CONTENT_LENGTH: -100")
print(f"handler.stdin.limit: {handler.stdin.limit}")
print(f"Initial stdin position: {handler.stdin._pos}")

# Try to read data
data = handler.stdin.read(10)
print(f"Attempted to read 10 bytes")
print(f"Bytes read: {len(data)}")
print(f"Data: {data!r}")

# Verify that the data was lost even though stdin had content
stdin.seek(0)  # Reset to check original content was there
original_data = stdin.read()
print(f"\nOriginal stdin content (still present): {original_data!r}")
print(f"Length of original content: {len(original_data)} bytes")
```

<details>

<summary>
Silent data loss - 23 bytes of valid request body completely unreadable
</summary>
```
CONTENT_LENGTH: -100
handler.stdin.limit: -100
Initial stdin position: 0
Attempted to read 10 bytes
Bytes read: 0
Data: b''

Original stdin content (still present): b'valid request body data'
Length of original content: 23 bytes
```
</details>

## Why This Is A Bug

This violates expected behavior in multiple critical ways:

1. **RFC 9110 Section 8.6 Violation**: The HTTP specification explicitly requires Content-Length to be a non-negative decimal integer. Negative values are invalid according to the standard and should be rejected or treated as 0.

2. **Silent Data Loss**: When a negative Content-Length is provided (whether through a malicious client, buggy proxy, or programming error), Django's LimitedStream silently discards all request body data. The condition in `LimitedStream.read()` at line 32 checks `if _pos >= limit`, which with `_pos=0` and `limit=-100` evaluates to `True`, immediately returning an empty byte string without ever reading the actual data.

3. **Security Implications**: This could be exploited to bypass request body validation. An attacker could manipulate proxies or send crafted requests with negative Content-Length headers to prevent Django from reading POST data, potentially circumventing CSRF checks, form validation, or other security mechanisms that depend on request body inspection.

4. **Inconsistent Error Handling**: While invalid non-numeric Content-Length values are safely converted to 0 (lines 126-127 in ServerHandler), negative numeric values pass through unchecked, creating an inconsistency in input validation.

## Relevant Context

The bug exists in Django's development server implementation (`django.core.servers.basehttp.ServerHandler`), which is used by `runserver` and `testserver` commands. The code comment at line 119-123 explicitly states this is "only for testserver/runserver", limiting the impact to development environments.

The same problematic pattern appears in `django.core.handlers.wsgi.WSGIRequest.__init__` (lines 74-77), suggesting this is a systematic issue in Django's WSGI handling:
- `/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/core/handlers/wsgi.py:75-77`
- `/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/core/servers/basehttp.py:124-127`

The LimitedStream class documentation references Werkzeug's implementation, but Werkzeug itself handles this case correctly by ensuring the limit is non-negative.

## Proposed Fix

```diff
--- a/django/core/servers/basehttp.py
+++ b/django/core/servers/basehttp.py
@@ -123,7 +123,10 @@ class ServerHandler(simple_server.ServerHandler):
         This fix applies only for testserver/runserver.
         """
         try:
             content_length = int(environ.get("CONTENT_LENGTH"))
+            # RFC 9110: Content-Length must be non-negative
+            if content_length < 0:
+                content_length = 0
         except (ValueError, TypeError):
             content_length = 0
         super().__init__(
```