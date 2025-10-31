# Bug Report: WSGIMiddleware SERVER_PORT Wrong Type

**Target**: `starlette.middleware.wsgi.build_environ`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `build_environ` function in the WSGI middleware sets `SERVER_PORT` as an integer instead of a string, violating the WSGI specification (PEP 3333) which requires it to be a string.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from starlette.middleware.wsgi import build_environ


@given(port=st.integers(min_value=1, max_value=65535))
@settings(max_examples=200)
def test_server_port_is_string_per_pep3333(port):
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

    assert isinstance(environ["SERVER_PORT"], str), \
        f"PEP 3333 requires SERVER_PORT to be a string, got {type(environ['SERVER_PORT'])}"
```

**Failing input**: Any valid port number (e.g., 8080)

## Reproducing the Bug

```python
from starlette.middleware.wsgi import build_environ

scope = {
    "type": "http",
    "method": "GET",
    "path": "/test",
    "query_string": b"",
    "headers": [],
    "server": ("example.com", 8080),
    "http_version": "1.1",
}

environ = build_environ(scope, b"")
print(f"SERVER_PORT: {environ['SERVER_PORT']!r}")
print(f"Type: {type(environ['SERVER_PORT'])}")
```

**Output**:
```
SERVER_PORT: 8080
Type: <class 'int'>
```

**Expected**:
```
SERVER_PORT: '8080'
Type: <class 'str'>
```

## Why This Is A Bug

PEP 3333 (WSGI specification) explicitly states:

> **SERVER_PORT**: The port portion of the server name, as a string.

WSGI applications and frameworks expect `SERVER_PORT` to be a string. While many may handle integers gracefully, this violates the specification and could cause issues with strict WSGI applications that expect string-based port comparisons or formatting.

Note: While `starlette.middleware.wsgi` is deprecated, it remains in the codebase and should conform to standards until removed.

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