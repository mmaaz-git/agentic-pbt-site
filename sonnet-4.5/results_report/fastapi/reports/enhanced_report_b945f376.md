# Bug Report: TrustedHostMiddleware IPv6 Host Header Parsing Failure

**Target**: `starlette.middleware.trustedhost.TrustedHostMiddleware`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

TrustedHostMiddleware fails to correctly parse IPv6 addresses from the Host header due to naive colon-based splitting, causing all IPv6 requests to be rejected with HTTP 400 even when the IPv6 address is explicitly included in allowed_hosts.

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

if __name__ == "__main__":
    test_host_header_with_ipv6_and_port()
```

<details>

<summary>
**Failing input**: `ipv6='::1', port=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/10/hypo.py", line 20, in <module>
    test_host_header_with_ipv6_and_port()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/10/hypo.py", line 11, in test_host_header_with_ipv6_and_port
    def test_host_header_with_ipv6_and_port(ipv6, port):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/10/hypo.py", line 16, in test_host_header_with_ipv6_and_port
    assert host == expected_host_without_port, \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: IPv6 host parsing is broken: [::1]:1 -> [
Falsifying example: test_host_header_with_ipv6_and_port(
    # The test always failed when commented parts were varied together.
    ipv6='::1',  # or any other generated value
    port=1,  # or any other generated value
)
```
</details>

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

# Test with IPv6 address in allowed_hosts
middleware = TrustedHostMiddleware(
    app=dummy_app,
    allowed_hosts=["[::1]"],
)

# Test with IPv6 host header including port
scope = {
    "type": "http",
    "method": "GET",
    "path": "/",
    "query_string": b"",
    "headers": [(b"host", b"[::1]:8000")],
}

asyncio.run(middleware(scope, dummy_receive, capture_send))

# Print the response status
if messages_sent:
    print(f"Response status: {messages_sent[0].get('status', 'N/A')}")
    if messages_sent[0].get('status') == 400:
        print("Error: IPv6 host was rejected despite being in allowed_hosts")
else:
    print("No response received")
```

<details>

<summary>
IPv6 host rejected with 400 despite being in allowed_hosts
</summary>
```
Response status: 400
Error: IPv6 host was rejected despite being in allowed_hosts
```
</details>

## Why This Is A Bug

The bug occurs in line 40 of `starlette/middleware/trustedhost.py`:
```python
host = headers.get("host", "").split(":")[0]
```

This parsing logic violates RFC 2732 and RFC 3986, which mandate that IPv6 addresses in URIs must be enclosed in square brackets. When a Host header contains `[::1]:8000`, the current implementation incorrectly extracts `[` instead of `[::1]` because it splits on the first colon, which appears within the IPv6 address itself.

The middleware then compares this malformed extraction (`[`) against the allowed_hosts list (containing `[::1]`), causing a mismatch and rejecting legitimate IPv6 requests with HTTP 400 "Invalid host header" even when the IPv6 address is explicitly allowed.

This makes the middleware completely non-functional in IPv6 environments, affecting both development (localhost ::1) and production deployments in IPv6-enabled infrastructure.

## Relevant Context

- **RFC 2732**: Specifies the format for literal IPv6 addresses in URLs as `[IPv6address]` or `[IPv6address]:port`
- **RFC 3986 Section 3.2.2**: Confirms IPv6 addresses must be enclosed in square brackets in the authority component
- **HTTP/1.1 Specification**: The Host header follows the same format as the URI authority component

The Starlette documentation doesn't explicitly mention IPv6 support, but as a modern web framework, it should handle standard-compliant HTTP headers. The current implementation works correctly for:
- IPv4 addresses: `127.0.0.1:8000` → `127.0.0.1` ✓
- Domain names: `example.com:8000` → `example.com` ✓
- IPv6 addresses: `[::1]:8000` → `[` ✗ (should be `[::1]`)

Code location: `/home/npc/miniconda/lib/python3.13/site-packages/starlette/middleware/trustedhost.py:40`

## Proposed Fix

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