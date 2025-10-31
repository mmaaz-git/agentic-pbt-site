# Bug Report: ServerHandler Negative Content-Length

**Target**: `django.core.servers.basehttp.ServerHandler.__init__`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

ServerHandler's CONTENT_LENGTH parsing handles invalid string values by defaulting to 0, but allows negative integers through, creating a LimitedStream with a negative limit instead of the intended 0.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.core.servers.basehttp import ServerHandler
from io import BytesIO

@given(st.integers())
def test_serverhandler_content_length_parsing_integers(content_length):
    stdin = BytesIO(b"test data")
    stdout = BytesIO()
    stderr = BytesIO()

    environ = {"CONTENT_LENGTH": str(content_length), "REQUEST_METHOD": "POST"}

    handler = ServerHandler(stdin, stdout, stderr, environ)

    expected_limit = max(0, content_length)
    assert handler.stdin.limit == expected_limit
```

**Failing input**: `content_length=-1`

## Reproducing the Bug

```python
from django.core.servers.basehttp import ServerHandler
from io import BytesIO

stdin = BytesIO(b"request body data")
stdout = BytesIO()
stderr = BytesIO()
environ = {"CONTENT_LENGTH": "-1", "REQUEST_METHOD": "POST"}

handler = ServerHandler(stdin, stdout, stderr, environ)

print(f"LimitedStream limit: {handler.stdin.limit}")
assert handler.stdin.limit == -1
```

## Why This Is A Bug

The ServerHandler code attempts to handle invalid CONTENT_LENGTH values by catching ValueError and TypeError and defaulting to 0. However, it doesn't handle the case where `int()` succeeds but produces a negative number. According to HTTP specifications, Content-Length must be non-negative. The code's intent is clear: normalize invalid values to 0. Negative integers violate this intent and create a LimitedStream with a semantically invalid negative limit.

## Fix

```diff
--- a/django/core/servers/basehttp.py
+++ b/django/core/servers/basehttp.py
@@ -124,7 +124,7 @@ class ServerHandler(simple_server.ServerHandler):
         try:
             content_length = int(environ.get("CONTENT_LENGTH"))
         except (ValueError, TypeError):
             content_length = 0
+        content_length = max(0, content_length)
         super().__init__(
             LimitedStream(stdin, content_length), stdout, stderr, environ, **kwargs
         )
```