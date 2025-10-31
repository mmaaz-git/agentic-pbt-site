# Bug Report: starlette.middleware.wsgi SERVER_PORT Type Violation

**Target**: `starlette.middleware.wsgi.build_environ`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

**Note**: This middleware is deprecated and will be removed in a future release.

## Summary

The WSGI middleware's `build_environ` function sets `SERVER_PORT` to an integer, violating PEP 3333 which requires it to be a string.

## Property-Based Test

```python
from hypothesis import given, settings
import hypothesis.strategies as st
from starlette.middleware.wsgi import build_environ


@given(st.integers(min_value=1, max_value=65535))
@settings(max_examples=100)
def test_server_port_type(port):
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "query_string": b"",
        "headers": [],
        "server": ("example.com", port),
        "http_version": "1.1",
    }

    environ = build_environ(scope, b"")

    assert isinstance(environ['SERVER_PORT'], str)
```

**Failing input**: Any port number (e.g., `port=1`)

## Reproducing the Bug

```python
from starlette.middleware.wsgi import build_environ


scope = {
    "type": "http",
    "method": "GET",
    "path": "/",
    "query_string": b"",
    "headers": [],
    "server": ("example.com", 8000),
    "http_version": "1.1",
}

environ = build_environ(scope, b"")

print(f"SERVER_PORT value: {repr(environ['SERVER_PORT'])}")
print(f"SERVER_PORT type: {type(environ['SERVER_PORT'])}")
```

Output:
```
SERVER_PORT value: 8000
SERVER_PORT type: <class 'int'>
```

## Why This Is A Bug

PEP 3333 (WSGI specification) section "environ Variables" states that `SERVER_PORT` must be a string:

> `SERVER_PORT`: The port portion of the URL, as a **string**.

The current implementation on line 50 of `wsgi.py` assigns the port directly:
```python
environ["SERVER_PORT"] = server[1]
```

This violates the WSGI spec and may cause compatibility issues with WSGI applications that expect a string.

## Fix

```diff
--- a/starlette/middleware/wsgi.py
+++ b/starlette/middleware/wsgi.py
@@ -47,7 +47,7 @@ def build_environ(scope: Scope, body: bytes) -> dict[str, Any]:
     # Get server name and port - required in WSGI, not in ASGI
     server = scope.get("server") or ("localhost", 80)
     environ["SERVER_NAME"] = server[0]
-    environ["SERVER_PORT"] = server[1]
+    environ["SERVER_PORT"] = str(server[1])

     # Get client IP address
     if scope.get("client"):
```