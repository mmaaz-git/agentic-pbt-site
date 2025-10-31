# Bug Report: HTTPSRedirectMiddleware Incorrectly Drops Port 443 from HTTP URLs

**Target**: `starlette.middleware.httpsredirect.HTTPSRedirectMiddleware`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

HTTPSRedirectMiddleware incorrectly drops port 443 when redirecting from HTTP to HTTPS, violating RFC 3986 URL normalization principles since port 443 is not the default port for HTTP.

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

    # This is the logic from HTTPSRedirectMiddleware line 14
    netloc = url.hostname if url.port in (80, 443) else url.netloc
    new_url = url.replace(scheme=redirect_scheme, netloc=netloc)

    # Assert that port 443 should be preserved when redirecting from HTTP
    assert ":443" in str(new_url), f"Port 443 should be preserved: {url} -> {new_url}"

if __name__ == "__main__":
    # Run the test
    test_http_port_443_redirect_preserves_port()
```

<details>

<summary>
<b>Failing input</b>: <code>hostname='aaa'</code>
</summary>

```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/3/hypo.py", line 27, in <module>
    test_http_port_443_redirect_preserves_port()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/3/hypo.py", line 5, in test_http_port_443_redirect_preserves_port
    def test_http_port_443_redirect_preserves_port(hostname):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/3/hypo.py", line 23, in test_http_port_443_redirect_preserves_port
    assert ":443" in str(new_url), f"Port 443 should be preserved: {url} -> {new_url}"
           ^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Port 443 should be preserved: http://aaa:443/test -> https://aaa/test
Falsifying example: test_http_port_443_redirect_preserves_port(
    hostname='aaa',  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
from starlette.datastructures import URL

# Test case demonstrating the bug
scope = {
    "type": "http",
    "scheme": "http",
    "server": ("example.com", 443),
    "path": "/test",
    "query_string": b"",
    "headers": [],
}

# Create URL from scope
url = URL(scope=scope)
print(f"Original URL: {url}")

# Simulate what HTTPSRedirectMiddleware does
redirect_scheme = {"http": "https", "ws": "wss"}[url.scheme]
netloc = url.hostname if url.port in (80, 443) else url.netloc
new_url = url.replace(scheme=redirect_scheme, netloc=netloc)

print(f"Redirected URL: {new_url}")
print(f"Expected URL: https://example.com:443/test")
print(f"")
print(f"Bug: Port 443 was dropped from the URL even though it's not the default port for HTTP")
print(f"     The middleware incorrectly treats port 443 as droppable for HTTP requests")
```

<details>

<summary>
Output showing port 443 incorrectly dropped
</summary>

```
Original URL: http://example.com:443/test
Redirected URL: https://example.com/test
Expected URL: https://example.com:443/test

Bug: Port 443 was dropped from the URL even though it's not the default port for HTTP
     The middleware incorrectly treats port 443 as droppable for HTTP requests
```
</details>

## Why This Is A Bug

This violates RFC 3986 URL normalization principles because the middleware incorrectly assumes port 443 can be dropped from HTTP URLs. According to RFC standards:

- **Port 80 is the default for HTTP** - can be omitted when the scheme is HTTP
- **Port 443 is the default for HTTPS** - can be omitted when the scheme is HTTPS
- **Port 443 is NOT a default for HTTP** - must be preserved when explicitly specified

The bug occurs at line 14 of `starlette/middleware/httpsredirect.py`:
```python
netloc = url.hostname if url.port in (80, 443) else url.netloc
```

This line incorrectly treats both ports 80 and 443 as droppable, regardless of the current scheme. When a client explicitly requests `http://example.com:443`, they are specifying a non-standard configuration (HTTP on port 443), and this explicit port choice should be preserved in the redirect.

Current incorrect behavior:
- `http://example.com:80` → `https://example.com` (correct - drops HTTP default)
- `http://example.com:443` → `https://example.com` (WRONG - should preserve non-default port)
- `http://example.com:8080` → `https://example.com:8080` (correct - preserves non-default port)

The same issue affects WebSocket redirects:
- `ws://example.com:443` → `wss://example.com` (WRONG - should preserve port 443)

## Relevant Context

This bug affects scenarios where:
- Development/testing environments use non-standard port configurations
- Proxy servers or load balancers use HTTP on port 443 internally
- Systems that explicitly configure HTTP on port 443 for specific reasons

The FastAPI middleware at `fastapi/middleware/httpsredirect.py` simply re-exports the Starlette implementation, so the bug affects both libraries.

Relevant documentation:
- [RFC 3986 Section 3.2.3](https://datatracker.ietf.org/doc/html/rfc3986#section-3.2.3) - URI normalization and default ports
- [Starlette HTTPSRedirectMiddleware source](https://github.com/encode/starlette/blob/master/starlette/middleware/httpsredirect.py)

## Proposed Fix

```diff
--- a/starlette/middleware/httpsredirect.py
+++ b/starlette/middleware/httpsredirect.py
@@ -11,7 +11,8 @@ class HTTPSRedirectMiddleware:
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