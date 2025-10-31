# Bug Report: TrustedHostMiddleware IPv6 Host Parsing

**Target**: `fastapi.middleware.trustedhost.TrustedHostMiddleware` (via `starlette.middleware.trustedhost`)
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

TrustedHostMiddleware incorrectly parses IPv6 addresses from the Host header, causing all IPv6 requests to be rejected even when the IPv6 address is in the allowed_hosts list.

## Property-Based Test

```python
from hypothesis import given, strategies as st

ipv6_addresses = st.sampled_from([
    "::1",
    "2001:db8::1",
    "fe80::1",
    "::ffff:192.0.2.1",
])

@given(ipv6_addresses, st.integers(min_value=1, max_value=65535))
def test_host_header_with_ipv6_and_port(ipv6, port):
    host_header_value = f"[{ipv6}]:{port}"
    host = host_header_value.split(":")[0]
    expected_host_without_port = f"[{ipv6}]"

    assert host == expected_host_without_port, \
        f"IPv6 host parsing is broken: {host_header_value} -> {host}"
```

**Failing input**: `ipv6='::1', port=8000`

## Reproducing the Bug

```python
import asyncio
from starlette.middleware.trustedhost import TrustedHostMiddleware

async def dummy_app(scope, receive, send):
    from starlette.responses import PlainTextResponse
    response = PlainTextResponse("OK")
    await response(scope, receive, send)

async def dummy_receive():
    return {"type": "http.request", "body": b"", "more_body": False}

messages_sent = []

async def capture_send(message):
    messages_sent.append(message)

middleware = TrustedHostMiddleware(
    app=dummy_app,
    allowed_hosts=["[::1]"],
)

scope = {
    "type": "http",
    "method": "GET",
    "path": "/",
    "query_string": b"",
    "headers": [(b"host", b"[::1]:8000")],
}

asyncio.run(middleware(scope, dummy_receive, capture_send))

print(messages_sent[0])
```

Output: `{'type': 'http.response.start', 'status': 400, ...}` (rejected)

## Why This Is A Bug

The bug is in line 40 of `starlette/middleware/trustedhost.py`:

```python
host = headers.get("host", "").split(":")[0]
```

This code attempts to remove the port by splitting on `:` and taking the first part. However, IPv6 addresses contain colons, so for a Host header like `[::1]:8000`:
- Expected result: `[::1]`
- Actual result: `[`

RFC 3986 specifies that IPv6 addresses in URIs must be enclosed in brackets, and when a port is present, it comes after the closing bracket: `[IPv6]:port`. The current parsing logic breaks this format.

## Fix

```diff
--- a/starlette/middleware/trustedhost.py
+++ b/starlette/middleware/trustedhost.py
@@ -37,7 +37,16 @@ class TrustedHostMiddleware:
             return

         headers = Headers(scope=scope)
-        host = headers.get("host", "").split(":")[0]
+        host_header = headers.get("host", "")
+
+        # Handle IPv6 addresses: [ipv6]:port or [ipv6]
+        if host_header.startswith("["):
+            # Find the closing bracket
+            bracket_end = host_header.find("]")
+            if bracket_end != -1:
+                host = host_header[:bracket_end + 1]
+            else:
+                host = host_header
+        else:
+            # IPv4 or hostname: split on first colon to remove port
+            host = host_header.split(":")[0]
         is_valid_host = False
         found_www_redirect = False
```