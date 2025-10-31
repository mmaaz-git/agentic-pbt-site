# Bug Report: Django Core Servers Negative CONTENT_LENGTH

**Target**: `django.core.servers.basehttp.ServerHandler.__init__` and `django.core.handlers.wsgi.WSGIRequest.__init__`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Django's development server accepts negative CONTENT_LENGTH values without validation, causing LimitedStream to block all request body reads due to its negative limit check (`_pos >= limit` evaluates to True when limit is negative).

## Property-Based Test

```python
from hypothesis import given, strategies as st
import io
from django.core.servers.basehttp import ServerHandler


@given(st.one_of(
    st.none(),
    st.text(),
    st.integers(),
    st.floats(allow_nan=True, allow_infinity=True),
    st.booleans(),
    st.lists(st.integers()),
))
def test_serverhandler_content_length_parsing(content_length_value):
    environ = {}
    if content_length_value is not None:
        environ["CONTENT_LENGTH"] = str(content_length_value)

    stdin = io.BytesIO(b"test data")
    stdout = io.BytesIO()
    stderr = io.BytesIO()

    handler = ServerHandler(stdin, stdout, stderr, environ)
    stream = handler.get_stdin()

    try:
        expected_length = int(content_length_value) if content_length_value is not None else 0
    except (ValueError, TypeError):
        expected_length = 0

    if expected_length < 0:
        expected_length = 0

    assert stream.limit == expected_length
```

**Failing input**: `content_length_value=-1`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

import io
from django.core.servers.basehttp import ServerHandler

environ = {"CONTENT_LENGTH": "-5"}
stdin = io.BytesIO(b"12345")
stdout = io.BytesIO()
stderr = io.BytesIO()

handler = ServerHandler(stdin, stdout, stderr, environ)
stream = handler.get_stdin()

print(f"CONTENT_LENGTH: {environ['CONTENT_LENGTH']}")
print(f"LimitedStream limit: {stream.limit}")

data = stream.read()
print(f"Data read: {data}")
print(f"Expected: b'12345', Actual: {data}")
```

Output:
```
CONTENT_LENGTH: -5
LimitedStream limit: -5
Data read: b''
Expected: b'12345', Actual: b''
```

## Why This Is A Bug

1. **HTTP Specification Violation**: RFC 9110 Section 8.6 specifies that Content-Length must be a non-negative decimal integer.

2. **Unexpected Behavior**: When CONTENT_LENGTH is negative, LimitedStream's `read()` method immediately returns empty bytes because `_pos >= limit` (0 >= -5) is True, blocking all request body access.

3. **Security Concern**: Malicious clients could exploit this by sending negative CONTENT_LENGTH to bypass request body processing, potentially causing application logic errors if the code expects to read POST data.

4. **Inconsistent with Error Handling**: The code properly handles non-numeric CONTENT_LENGTH by defaulting to 0, but fails to validate that parsed integers are non-negative.

## Fix

```diff
--- a/django/core/servers/basehttp.py
+++ b/django/core/servers/basehttp.py
@@ -124,7 +124,10 @@ class ServerHandler(simple_server.ServerHandler):
         try:
             content_length = int(environ.get("CONTENT_LENGTH"))
         except (ValueError, TypeError):
             content_length = 0
+        if content_length < 0:
+            content_length = 0
+
         super().__init__(
             LimitedStream(stdin, content_length), stdout, stderr, environ, **kwargs
         )
```

And the same fix should be applied to WSGIRequest:

```diff
--- a/django/core/handlers/wsgi.py
+++ b/django/core/handlers/wsgi.py
@@ -74,6 +74,8 @@ class WSGIRequest(HttpRequest):
         try:
             content_length = int(environ.get("CONTENT_LENGTH"))
         except (ValueError, TypeError):
             content_length = 0
+        if content_length < 0:
+            content_length = 0
         self._stream = LimitedStream(self.environ["wsgi.input"], content_length)
         self._read_started = False
```