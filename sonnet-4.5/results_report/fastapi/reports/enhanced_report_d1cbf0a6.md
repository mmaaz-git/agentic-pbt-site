# Bug Report: HTTPSRedirectMiddleware Incorrectly Strips Non-Standard HTTP Port 443

**Target**: `starlette.middleware.httpsredirect.HTTPSRedirectMiddleware`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The HTTPSRedirectMiddleware incorrectly strips port 443 when redirecting from HTTP to HTTPS, treating it as a standard port for HTTP when it is actually only standard for HTTPS. This causes HTTP services running on port 443 to lose their port specification during redirect.

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

    # This is the exact logic from HTTPSRedirectMiddleware
    netloc = url.hostname if url.port in (80, 443) else url.netloc
    result_url = url.replace(scheme="https", netloc=netloc)

    assert ":443" in str(result_url), \
        f"Port 443 should be preserved when redirecting http://{hostname}:443 to https, " \
        f"but got {result_url}. Port 443 is non-standard for HTTP."


if __name__ == "__main__":
    test_http_on_port_443_loses_port_on_redirect()
```

<details>

<summary>
**Failing input**: `hostname='aaa'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/14/hypo.py", line 28, in <module>
    test_http_on_port_443_loses_port_on_redirect()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/14/hypo.py", line 6, in test_http_on_port_443_loses_port_on_redirect
    def test_http_on_port_443_loses_port_on_redirect(hostname):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/14/hypo.py", line 22, in test_http_on_port_443_loses_port_on_redirect
    assert ":443" in str(result_url), \
           ^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Port 443 should be preserved when redirecting http://aaa:443 to https, but got https://aaa/. Port 443 is non-standard for HTTP.
Falsifying example: test_http_on_port_443_loses_port_on_redirect(
    hostname='aaa',  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
from starlette.datastructures import URL

# Create a scope representing HTTP on port 443
scope = {
    "type": "http",
    "scheme": "http",
    "server": ("example.com", 443),
    "path": "/test",
    "query_string": b"",
    "headers": []
}

# Create URL from scope
url = URL(scope=scope)
print(f"Original URL: {url}")

# This is the exact logic from HTTPSRedirectMiddleware line 13-15
redirect_scheme = {"http": "https", "ws": "wss"}[url.scheme]
netloc = url.hostname if url.port in (80, 443) else url.netloc
result_url = url.replace(scheme=redirect_scheme, netloc=netloc)

print(f"Redirect URL: {result_url}")
print(f"\nProblem: Port 443 was stripped even though it's non-standard for HTTP!")
print(f"Expected: https://example.com:443/test")
print(f"Actual:   {result_url}")
```

<details>

<summary>
Demonstrates incorrect port stripping for HTTP on port 443
</summary>
```
Original URL: http://example.com:443/test
Redirect URL: https://example.com/test

Problem: Port 443 was stripped even though it's non-standard for HTTP!
Expected: https://example.com:443/test
Actual:   https://example.com/test
```
</details>

## Why This Is A Bug

The bug occurs in line 14 of `starlette/middleware/httpsredirect.py` where the middleware checks `if url.port in (80, 443)` to decide whether to strip the port. This violates standard URL conventions because:

1. **Port 80 is the standard port for HTTP** - Stripping it when redirecting from `http://example.com:80` to HTTPS is correct behavior
2. **Port 443 is the standard port for HTTPS, not HTTP** - When an HTTP service runs on port 443 (non-standard), the port should be preserved during redirect

According to RFC 7230 and RFC 1700, standard ports are:
- HTTP: Port 80
- HTTPS: Port 443
- WS: Port 80 (same as HTTP)
- WSS: Port 443 (same as HTTPS)

The middleware incorrectly treats port 443 as "always strippable" regardless of the source scheme. This causes HTTP services running on port 443 to be redirected to `https://example.com` (default HTTPS port 443) instead of `https://example.com:443`, potentially reaching a different service or failing to connect entirely.

## Relevant Context

The bug affects both HTTP and WebSocket redirects since the same logic applies to WS→WSS redirections. The issue manifests in these scenarios:
- `http://example.com:443` → `https://example.com` (WRONG - should preserve :443)
- `ws://example.com:443` → `wss://example.com` (WRONG - should preserve :443)
- `http://example.com:80` → `https://example.com` (CORRECT - standard HTTP port)
- `http://example.com:8080` → `https://example.com:8080` (CORRECT - non-standard port preserved)

While running HTTP on port 443 is uncommon, it's a valid configuration that may occur in:
- Development environments with port constraints
- Reverse proxy configurations
- Services that need to appear on port 443 but don't use TLS
- Migration scenarios where services are being moved between protocols

The Starlette source code is at: https://github.com/encode/starlette/blob/master/starlette/middleware/httpsredirect.py

## Proposed Fix

```diff
--- a/starlette/middleware/httpsredirect.py
+++ b/starlette/middleware/httpsredirect.py
@@ -11,7 +11,8 @@ class HTTPSRedirectMiddleware:
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