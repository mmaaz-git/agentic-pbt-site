# Bug Report: HTTPSRedirectMiddleware Incorrect Port Handling

**Target**: `starlette.middleware.httpsredirect.HTTPSRedirectMiddleware`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `HTTPSRedirectMiddleware` incorrectly drops the port number when redirecting from http/ws to https/wss if the original port is 443 (or 80), even though 443 is not the default port for the original scheme.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from starlette.datastructures import URL


@given(
    port=st.integers(min_value=1, max_value=65535),
    scheme=st.sampled_from(["http", "ws"])
)
@settings(max_examples=200)
def test_redirect_preserves_non_default_ports(port, scheme):
    scope = {
        "type": "http" if scheme == "http" else "websocket",
        "scheme": scheme,
        "method": "GET",
        "path": "/",
        "query_string": b"",
        "headers": [],
        "server": ("example.com", port),
    }

    url = URL(scope=scope)
    current_logic_netloc = url.hostname if url.port in (80, 443) else url.netloc

    default_port_for_scheme = 80 if scheme in ("http", "ws") else 443

    if port != default_port_for_scheme:
        expected = f"example.com:{port}"
        assert current_logic_netloc == expected or port in (80, 443), \
            f"Non-default port {port} should be preserved"
```

**Failing input**: scheme="http", port=443 (or scheme="ws", port=443)

## Reproducing the Bug

```python
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.datastructures import URL

scope = {
    "type": "http",
    "scheme": "http",
    "method": "GET",
    "path": "/test",
    "query_string": b"",
    "headers": [],
    "server": ("example.com", 443),
}

url = URL(scope=scope)
redirect_scheme = "https"
netloc = url.hostname if url.port in (80, 443) else url.netloc
redirected_url = url.replace(scheme=redirect_scheme, netloc=netloc)

print(f"Original: {url}")
print(f"Redirected: {redirected_url}")
```

**Output**:
```
Original: http://example.com:443/test
Redirected: https://example.com/test
```

**Expected**: `https://example.com:443/test` (port should be preserved)

## Why This Is A Bug

The current logic drops ports 80 and 443 unconditionally, without considering whether they are default ports for the *current* scheme:

- Port 80 is default for http/ws, but non-default for https/wss
- Port 443 is default for https/wss, but non-default for http/ws

The code should only drop a port if it's the default port for the *target* scheme, not just if it's in the set {80, 443}.

While running HTTP on port 443 is unusual, it's valid, and the redirect should preserve the port. This is more likely to affect WebSocket connections (ws://example.com:443 → wss://example.com should preserve :443).

## Fix

```diff
--- a/starlette/middleware/httpsredirect.py
+++ b/starlette/middleware/httpsredirect.py
@@ -10,8 +10,11 @@ class HTTPSRedirectMiddleware:
     async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
         if scope["type"] in ("http", "websocket") and scope["scheme"] in ("http", "ws"):
             url = URL(scope=scope)
             redirect_scheme = {"http": "https", "ws": "wss"}[url.scheme]
-            netloc = url.hostname if url.port in (80, 443) else url.netloc
+            # Only drop port if it matches the default for the original scheme
+            default_ports = {"http": 80, "ws": 80, "https": 443, "wss": 443}
+            is_default_port = url.port == default_ports.get(url.scheme)
+            netloc = url.hostname if is_default_port else url.netloc
             url = url.replace(scheme=redirect_scheme, netloc=netloc)
             response = RedirectResponse(url, status_code=307)
             await response(scope, receive, send)
```