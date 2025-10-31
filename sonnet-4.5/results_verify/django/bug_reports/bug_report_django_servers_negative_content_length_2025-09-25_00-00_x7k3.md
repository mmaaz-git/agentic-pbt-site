# Bug Report: Django ServerHandler Negative Content-Length

**Target**: `django.core.servers.basehttp.ServerHandler.__init__`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `ServerHandler` class accepts negative `CONTENT_LENGTH` values from HTTP headers, which causes the `LimitedStream` to become completely unreadable, preventing valid request body data from being processed.

## Property-Based Test

```python
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
```

**Failing input**: `CONTENT_LENGTH="-100"`

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

environ = {"CONTENT_LENGTH": "-100"}
stdin = io.BytesIO(b"valid request body data")
stdout = io.BytesIO()
stderr = io.BytesIO()

handler = ServerHandler(stdin, stdout, stderr, environ)

print(f"CONTENT_LENGTH: -100")
print(f"handler.stdin.limit: {handler.stdin.limit}")

data = handler.stdin.read(10)
print(f"Bytes read: {len(data)}")
print(f"Data: {data}")
```

**Output:**
```
CONTENT_LENGTH: -100
handler.stdin.limit: -100
Bytes read: 0
Data: b''
```

## Why This Is A Bug

1. **HTTP Specification Violation**: RFC 9110 Section 8.6 states that `Content-Length` must be a non-negative integer. Negative values are invalid.

2. **Silent Data Loss**: When a negative content-length is provided (maliciously or by a buggy proxy), the `LimitedStream` silently discards all request body data because the condition `_pos >= limit` (i.e., `0 >= -100`) is immediately true.

3. **Inconsistent with Django's own code**: In `django/core/handlers/wsgi.py:75-77`, the same parsing logic is used, demonstrating this is a systemic issue across Django's WSGI handling.

4. **Security concern**: A malicious proxy or client could inject negative `Content-Length` headers to bypass request body processing, potentially circumventing security checks that inspect POST data.

## Fix

```diff
--- a/django/core/servers/basehttp.py
+++ b/django/core/servers/basehttp.py
@@ -123,7 +123,7 @@ class ServerHandler(simple_server.ServerHandler):
         shouldn't discard the data since the upstream servers usually do this.
         This fix applies only for testserver/runserver.
         """
         try:
-            content_length = int(environ.get("CONTENT_LENGTH"))
+            content_length = max(0, int(environ.get("CONTENT_LENGTH")))
         except (ValueError, TypeError):
             content_length = 0
         super().__init__(
```

Alternatively, for better error reporting:

```diff
--- a/django/core/servers/basehttp.py
+++ b/django/core/servers/basehttp.py
@@ -123,8 +123,10 @@ class ServerHandler(simple_server.ServerHandler):
         shouldn't discard the data since the upstream servers usually do this.
         This fix applies only for testserver/runserver.
         """
         try:
             content_length = int(environ.get("CONTENT_LENGTH"))
+            if content_length < 0:
+                content_length = 0
         except (ValueError, TypeError):
             content_length = 0
         super().__init__(
```

Note: The same fix should be applied to `django/core/handlers/wsgi.py:75-77` for consistency.