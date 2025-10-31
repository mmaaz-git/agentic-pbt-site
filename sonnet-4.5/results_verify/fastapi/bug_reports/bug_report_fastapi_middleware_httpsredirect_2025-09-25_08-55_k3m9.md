# Bug Report: HTTPSRedirectMiddleware Incorrect Port Handling

**Target**: `starlette.middleware.httpsredirect.HTTPSRedirectMiddleware` (re-exported as `fastapi.middleware.httpsredirect.HTTPSRedirectMiddleware`)
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

HTTPSRedirectMiddleware incorrectly drops port 443 when redirecting from HTTP to HTTPS, even when port 443 was explicitly specified for HTTP (a non-standard but valid configuration).

## Property-Based Test

```python
from hypothesis import given, strategies as st
from starlette.datastructures import URL

@given(st.text(alphabet=st.characters(whitelist_categories=('Ll',), min_codepoint=97, max_codepoint=122), min_size=3, max_size=15))
def test_http_port_443_redirect_preserves_port(hostname):
    scope = {
        "type": "http",
        "scheme": "http",
        "server": (hostname, 443),
        "path": "/test",
        "query_string": b"",
        "headers": [],
    }

    url = URL(scope=scope)
    redirect_scheme = {"http": "https", "ws": "wss"}[url.scheme]

    netloc = url.hostname if url.port in (80, 443) else url.netloc
    new_url = url.replace(scheme=redirect_scheme, netloc=netloc)

    assert ":443" in str(new_url), f"Port 443 should be preserved: {url} -> {new_url}"
```

**Failing input**: Any hostname with HTTP on port 443, e.g., `http://example.com:443`

## Reproducing the Bug

```python
from starlette.datastructures import URL

scope = {
    "type": "http",
    "scheme": "http",
    "server": ("example.com", 443),
    "path": "/test",
    "query_string": b"",
    "headers": [],
}

url = URL(scope=scope)
redirect_scheme = {"http": "https", "ws": "wss"}[url.scheme]
netloc = url.hostname if url.port in (80, 443) else url.netloc
new_url = url.replace(scheme=redirect_scheme, netloc=netloc)

print(f"Original: {url}")
print(f"Redirected: {new_url}")
print(f"Expected: https://example.com:443")
print(f"Bug: Port 443 was dropped")
```

**Output:**
```
Original: http://example.com:443/test
Redirected: https://example.com/test
Expected: https://example.com:443/test
Bug: Port 443 was dropped
```

## Why This Is A Bug

When a client accesses `http://example.com:443`, they are explicitly requesting HTTP protocol on port 443 (non-standard, as HTTP typically uses port 80). The HTTPSRedirectMiddleware should preserve this explicit port choice when redirecting to HTTPS.

The current code (line 14 of `httpsredirect.py`):
```python
netloc = url.hostname if url.port in (80, 443) else url.netloc
```

This logic assumes both port 80 and 443 can be dropped, but:
- **Port 80**: The HTTP default port, reasonable to drop when redirecting to HTTPS
- **Port 443**: NOT the HTTP default port; when explicitly specified for HTTP, it should be preserved

**Correct behavior:**
- `http://example.com:80` → `https://example.com` ✓ (drop HTTP default port)
- `http://example.com:443` → `https://example.com:443` ✗ (preserve explicit non-default port)
- `http://example.com:8080` → `https://example.com:8080` ✓ (preserve explicit port)

**Similar issue for WebSocket:**
- `ws://example.com:443` → `wss://example.com` ✗ (should preserve port 443)

## Fix

Only drop the port if it matches the default port for the *current* scheme (before redirect), not the target scheme:

```diff
--- a/starlette/middleware/httpsredirect.py
+++ b/starlette/middleware/httpsredirect.py
@@ -11,7 +11,8 @@ class HTTPSRedirectMiddleware:
     async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
         if scope["type"] in ("http", "websocket") and scope["scheme"] in ("http", "ws"):
             url = URL(scope=scope)
             redirect_scheme = {"http": "https", "ws": "wss"}[url.scheme]
-            netloc = url.hostname if url.port in (80, 443) else url.netloc
+            default_port = {"http": 80, "ws": 80}[url.scheme]
+            netloc = url.hostname if url.port == default_port else url.netloc
             url = url.replace(scheme=redirect_scheme, netloc=netloc)
             response = RedirectResponse(url, status_code=307)
             await response(scope, receive, send)
```