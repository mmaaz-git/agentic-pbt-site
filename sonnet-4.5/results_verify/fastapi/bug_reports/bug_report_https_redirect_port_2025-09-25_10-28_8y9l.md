# Bug Report: HTTPSRedirectMiddleware Incorrect Port Stripping

**Target**: `fastapi.middleware.httpsredirect.HTTPSRedirectMiddleware` (via `starlette.middleware.httpsredirect.HTTPSRedirectMiddleware`)
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The HTTPSRedirectMiddleware incorrectly strips port 443 from URLs when redirecting from HTTP to HTTPS, even though port 443 is a non-standard port for HTTP. This causes HTTP services running on port 443 to lose their port number when redirected to HTTPS.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from starlette.datastructures import URL


@given(hostname=st.text(alphabet=st.characters(min_codepoint=97, max_codepoint=122), min_size=3, max_size=15))
def test_http_on_port_443_loses_port_on_redirect(hostname):
    scope = {
        "type": "http",
        "scheme": "http",
        "server": (hostname, 443),
        "path": "/",
        "query_string": b"",
        "headers": []
    }

    url = URL(scope=scope)

    netloc = url.hostname if url.port in (80, 443) else url.netloc
    result_url = url.replace(scheme="https", netloc=netloc)

    assert ":443" in str(result_url), \
        f"Port 443 should be preserved when redirecting http://{hostname}:443 to https, " \
        f"but got {result_url}. Port 443 is non-standard for HTTP."
```

**Failing input**: Any hostname (e.g., `hostname='example'`)

## Reproducing the Bug

```python
from starlette.datastructures import URL

scope = {
    "type": "http",
    "scheme": "http",
    "server": ("example.com", 443),
    "path": "/test",
    "query_string": b"",
    "headers": []
}

url = URL(scope=scope)

redirect_scheme = "https"
netloc = url.hostname if url.port in (80, 443) else url.netloc
result_url = url.replace(scheme=redirect_scheme, netloc=netloc)

print(f"Original URL: {url}")
print(f"Redirect URL: {result_url}")
```

Output:
```
Original URL: http://example.com:443/test
Redirect URL: https://example.com/test
```

## Why This Is A Bug

The logic in line 14 of `starlette/middleware/httpsredirect.py` checks if the port is 80 or 443:

```python
netloc = url.hostname if url.port in (80, 443) else url.netloc
```

This is incorrect because:

1. **Port 80 is the standard port for HTTP** - Stripping it when redirecting `http://example.com:80` → `https://example.com` is arguably correct (though debatable)

2. **Port 443 is the standard port for HTTPS, not HTTP** - When an HTTP service runs on port 443 (non-standard), the redirect should preserve the port: `http://example.com:443` → `https://example.com:443`

The current implementation treats port 443 as "always strippable" regardless of the source scheme, which is wrong.

**Real-world impact**: If a user configures an HTTP server on port 443 and uses this middleware, clients will be redirected to `https://example.com` (default port 443) instead of `https://example.com:443`, potentially reaching a different service or failing entirely.

## Fix

The fix should check if the port is the **standard port for the current scheme**, not a hardcoded list:

```diff
--- a/starlette/middleware/httpsredirect.py
+++ b/starlette/middleware/httpsredirect.py
@@ -10,7 +10,8 @@ class HTTPSRedirectMiddleware:
     async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
         if scope["type"] in ("http", "websocket") and scope["scheme"] in ("http", "ws"):
             url = URL(scope=scope)
             redirect_scheme = {"http": "https", "ws": "wss"}[url.scheme]
-            netloc = url.hostname if url.port in (80, 443) else url.netloc
+            standard_port = {"http": 80, "ws": 80}[url.scheme]
+            netloc = url.hostname if url.port == standard_port else url.netloc
             url = url.replace(scheme=redirect_scheme, netloc=netloc)
             response = RedirectResponse(url, status_code=307)
             await response(scope, receive, send)
```