# Bug Report: TrustedHostMiddleware Fails to Parse IPv6 Host Headers

**Target**: `starlette.middleware.trustedhost.TrustedHostMiddleware.__call__`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

TrustedHostMiddleware incorrectly parses Host headers containing IPv6 addresses, causing all IPv6 requests to be rejected with 400 Bad Request even when the IPv6 address is explicitly allowed in the allowed_hosts configuration.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.datastructures import Headers


@given(
    st.integers(min_value=1, max_value=65535)
)
def test_ipv6_host_parsing(port):
    class MockApp:
        async def __call__(self, scope, receive, send):
            pass

    middleware = TrustedHostMiddleware(
        MockApp(),
        allowed_hosts=["[::1]"]
    )

    host_with_port = f"[::1]:{port}"
    headers = Headers({"host": host_with_port})

    host = headers.get("host", "").split(":")[0]

    assert host == "[::1]", \
        f"Expected '[::1]' but got '{host}' when parsing '{host_with_port}'"


if __name__ == "__main__":
    test_ipv6_host_parsing()
```

<details>

<summary>
**Failing input**: `port=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/8/hypo.py", line 29, in <module>
    test_ipv6_host_parsing()
    ~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/8/hypo.py", line 7, in test_ipv6_host_parsing
    st.integers(min_value=1, max_value=65535)
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/8/hypo.py", line 24, in test_ipv6_host_parsing
    assert host == "[::1]", \
           ^^^^^^^^^^^^^^^
AssertionError: Expected '[::1]' but got '[' when parsing '[::1]:1'
Falsifying example: test_ipv6_host_parsing(
    port=1,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import asyncio
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.datastructures import Headers

async def test_ipv6_parsing():
    """Demonstrate TrustedHostMiddleware IPv6 parsing bug"""

    # Mock app that just returns 200 OK
    class MockApp:
        async def __call__(self, scope, receive, send):
            await send({
                'type': 'http.response.start',
                'status': 200,
                'headers': [[b'content-type', b'text/plain']],
            })
            await send({
                'type': 'http.response.body',
                'body': b'OK',
            })

    # Create middleware that should allow IPv6 localhost
    middleware = TrustedHostMiddleware(
        MockApp(),
        allowed_hosts=["[::1]", "[2001:db8::1]", "localhost"]
    )

    # Test various Host header formats
    test_cases = [
        ("[::1]", "IPv6 localhost"),
        ("[::1]:8080", "IPv6 localhost with port"),
        ("[2001:db8::1]", "IPv6 address"),
        ("[2001:db8::1]:8080", "IPv6 address with port"),
        ("localhost", "Regular hostname"),
        ("localhost:8080", "Regular hostname with port"),
    ]

    print("Testing TrustedHostMiddleware with IPv6 addresses:")
    print("=" * 60)
    print(f"Allowed hosts: {middleware.allowed_hosts}")
    print()

    for host_header, description in test_cases:
        # Show what the current parsing logic does
        parsed_host = host_header.split(":")[0]
        print(f"Test: {description}")
        print(f"  Host header: '{host_header}'")
        print(f"  Parsed host (using split(':')[0]): '{parsed_host}'")

        # Create a mock HTTP scope
        scope = {
            'type': 'http',
            'method': 'GET',
            'path': '/',
            'headers': [(b'host', host_header.encode())],
            'query_string': b'',
            'root_path': '',
            'scheme': 'http',
            'server': ('127.0.0.1', 8000),
        }

        # Track if response was sent
        response_sent = False
        response_status = None

        async def receive():
            return {'type': 'http.request', 'body': b''}

        async def send(message):
            nonlocal response_sent, response_status
            if message['type'] == 'http.response.start':
                response_sent = True
                response_status = message['status']

        # Test the middleware
        await middleware(scope, receive, send)

        if response_status == 200:
            print(f"  Result: ✓ ALLOWED (200 OK)")
        else:
            print(f"  Result: ✗ REJECTED ({response_status} - Should be allowed!)")

        # Show why it fails for IPv6
        if host_header.startswith("["):
            print(f"  Issue: IPv6 address incorrectly parsed as '{parsed_host}'")
            if parsed_host in middleware.allowed_hosts:
                print(f"         '{parsed_host}' IS in allowed_hosts")
            else:
                print(f"         '{parsed_host}' NOT in allowed_hosts (that's why it fails!)")
        print()

# Run the test
asyncio.run(test_ipv6_parsing())
```

<details>

<summary>
TrustedHostMiddleware rejects all IPv6 addresses even when explicitly allowed
</summary>
```
Testing TrustedHostMiddleware with IPv6 addresses:
============================================================
Allowed hosts: ['[::1]', '[2001:db8::1]', 'localhost']

Test: IPv6 localhost
  Host header: '[::1]'
  Parsed host (using split(':')[0]): '['
  Result: ✗ REJECTED (400 - Should be allowed!)
  Issue: IPv6 address incorrectly parsed as '['
         '[' NOT in allowed_hosts (that's why it fails!)

Test: IPv6 localhost with port
  Host header: '[::1]:8080'
  Parsed host (using split(':')[0]): '['
  Result: ✗ REJECTED (400 - Should be allowed!)
  Issue: IPv6 address incorrectly parsed as '['
         '[' NOT in allowed_hosts (that's why it fails!)

Test: IPv6 address
  Host header: '[2001:db8::1]'
  Parsed host (using split(':')[0]): '[2001'
  Result: ✗ REJECTED (400 - Should be allowed!)
  Issue: IPv6 address incorrectly parsed as '[2001'
         '[2001' NOT in allowed_hosts (that's why it fails!)

Test: IPv6 address with port
  Host header: '[2001:db8::1]:8080'
  Parsed host (using split(':')[0]): '[2001'
  Result: ✗ REJECTED (400 - Should be allowed!)
  Issue: IPv6 address incorrectly parsed as '[2001'
         '[2001' NOT in allowed_hosts (that's why it fails!)

Test: Regular hostname
  Host header: 'localhost'
  Parsed host (using split(':')[0]): 'localhost'
  Result: ✓ ALLOWED (200 OK)

Test: Regular hostname with port
  Host header: 'localhost:8080'
  Parsed host (using split(':')[0]): 'localhost'
  Result: ✓ ALLOWED (200 OK)

```
</details>

## Why This Is A Bug

The bug violates RFC 3986 Section 3.2.2, which mandates that IPv6 addresses in URIs must be enclosed in square brackets. The HTTP Host header follows this same format specification. The middleware's parsing logic at line 40 (`host = headers.get("host", "").split(":")[0]`) fails because IPv6 addresses contain colons as part of their syntax. When parsing `[::1]:8080`, the code splits on the first colon inside the IPv6 address, returning `[` instead of the complete `[::1]`. This makes IPv6 support completely broken - even when IPv6 addresses are explicitly added to `allowed_hosts`, they are always rejected because the malformed parsed value never matches. This is a critical issue for modern deployments where IPv6 is increasingly common in cloud environments, container orchestration systems, and mobile networks.

## Relevant Context

The bug is located at line 40 of `/home/npc/pbt/agentic-pbt/envs/fastapi_env/lib/python3.13/site-packages/starlette/middleware/trustedhost.py`. The middleware is also re-exported by FastAPI, affecting all FastAPI applications using this security feature.

IPv6 addresses in Host headers follow the standardized format defined in RFC 3986:
- Without port: `[::1]` or `[2001:db8::1]`
- With port: `[::1]:8080` or `[2001:db8::1]:443`

The square brackets distinguish the colons in the IPv6 address from the port separator colon. This is universally adopted by all major web servers, browsers, and HTTP clients. The middleware's failure to handle this standard format means it cannot be used in:
- IPv6-only environments (increasingly common in modern infrastructure)
- Dual-stack deployments where clients may use either IPv4 or IPv6
- Local development with IPv6 loopback addresses
- Container environments where IPv6 is preferred for network isolation

## Proposed Fix

```diff
--- a/starlette/middleware/trustedhost.py
+++ b/starlette/middleware/trustedhost.py
@@ -37,7 +37,15 @@ class TrustedHostMiddleware:
             return

         headers = Headers(scope=scope)
-        host = headers.get("host", "").split(":")[0]
+        host_header = headers.get("host", "")
+
+        # Handle IPv6 addresses in brackets, e.g., [::1]:port or [::1]
+        if host_header.startswith("["):
+            # IPv6 address: extract everything up to and including the closing bracket
+            bracket_end = host_header.find("]")
+            host = host_header[:bracket_end + 1] if bracket_end != -1 else host_header
+        else:
+            # IPv4 or hostname: split by colon to remove port
+            host = host_header.split(":")[0]
         is_valid_host = False
         found_www_redirect = False
         for pattern in self.allowed_hosts:
```