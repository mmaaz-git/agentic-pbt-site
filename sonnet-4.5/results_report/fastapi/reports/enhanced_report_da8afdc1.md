# Bug Report: TrustedHostMiddleware IPv6 Address Parsing Failure

**Target**: `starlette.middleware.trustedhost.TrustedHostMiddleware`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

TrustedHostMiddleware incorrectly parses IPv6 addresses in the Host header by splitting on the first colon, causing all IPv6 requests to be rejected with 400 status codes even when the IPv6 address is explicitly allowed in the allowed_hosts list.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Property-based test for TrustedHostMiddleware IPv6 parsing bug."""

from hypothesis import given, strategies as st, example

@given(
    st.lists(st.integers(min_value=0, max_value=65535).map(lambda x: hex(x)[2:]), min_size=8, max_size=8),
    st.integers(min_value=1, max_value=65535)
)
@example(segments=['2', '0', '0', '1', 'd', 'b', '8', '1'], port=8080)
def test_trustedhost_ipv6_parsing(segments, port):
    ipv6 = ':'.join(segments)
    host_with_port = f"[{ipv6}]:{port}"

    parsed = host_with_port.split(":")[0]
    expected = f"[{ipv6}]"

    assert parsed == expected or parsed == ipv6

if __name__ == "__main__":
    # Run the full property-based test
    print("Running Hypothesis property-based test for IPv6 parsing bug...")
    print("This test verifies that splitting IPv6 addresses with ports by ':' breaks parsing")
    print("=" * 70)

    try:
        test_trustedhost_ipv6_parsing()
        print("\nAll tests passed!")
    except Exception as e:
        print(f"\nTest suite failed with error:")
        print(str(e))
```

<details>

<summary>
**Failing input**: `segments=['2', '0', '0', '1', 'd', 'b', '8', '1'], port=8080`
</summary>
```
Traceback (most recent call last):
  File "<string>", line 17, in <module>
    test_trustedhost_ipv6_parsing()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "<string>", line 4, in test_trustedhost_ipv6_parsing
    st.lists(st.integers(min_value=0, max_value=65535).map(lambda x: hex(x)[2:]), min_size=8, max_size=8),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
    _raise_to_user(errors, state.settings, [], " in explicit examples")
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
    raise the_error_hypothesis_found
  File "<string>", line 15, in test_trustedhost_ipv6_parsing
    assert parsed == expected or parsed == ipv6, f'parsed={parsed}, expected={expected} or {ipv6}'
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: parsed=[2, expected=[2:0:0:1:d:b:8:1] or 2:0:0:1:d:b:8:1
Falsifying explicit example: test_trustedhost_ipv6_parsing(
    segments=['2', '0', '0', '1', 'd', 'b', '8', '1'],
    port=8080,
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction of TrustedHostMiddleware IPv6 parsing bug."""

from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import PlainTextResponse
from starlette.testclient import TestClient

async def app(scope, receive, send):
    """Simple app that returns success."""
    response = PlainTextResponse("Request accepted", status_code=200)
    await response(scope, receive, send)

# Create middleware with IPv6 address in allowed_hosts
middleware = TrustedHostMiddleware(
    app,
    allowed_hosts=["[2001:db8::1]", "2001:db8::1", "localhost"]
)

# Test with TestClient
client = TestClient(middleware)

print("Testing TrustedHostMiddleware with IPv6 addresses")
print("=" * 50)

# Test 1: localhost (should work)
print("\nTest 1: Request to localhost")
response = client.get("/", headers={"host": "localhost"})
print(f"  Host header: localhost")
print(f"  Status code: {response.status_code}")
print(f"  Response: {response.text}")

# Test 2: IPv6 with port (demonstrates the bug)
print("\nTest 2: Request to IPv6 address with port")
response = client.get("/", headers={"host": "[2001:db8::1]:8080"})
print(f"  Host header: [2001:db8::1]:8080")
print(f"  Status code: {response.status_code}")
print(f"  Response: {response.text}")

# Test 3: IPv6 without port
print("\nTest 3: Request to IPv6 address without port")
response = client.get("/", headers={"host": "[2001:db8::1]"})
print(f"  Host header: [2001:db8::1]")
print(f"  Status code: {response.status_code}")
print(f"  Response: {response.text}")

# Demonstrate the parsing issue directly
print("\n" + "=" * 50)
print("Direct parsing demonstration:")
host_with_port = "[2001:db8::1]:8080"
parsed = host_with_port.split(":")[0]
print(f"  Original: {host_with_port}")
print(f"  After split(':')[0]: {parsed}")
print(f"  Expected: [2001:db8::1] or 2001:db8::1")
print(f"  Result: The parsed value '{parsed}' will not match any allowed_hosts")
```

<details>

<summary>
TrustedHostMiddleware rejects valid IPv6 addresses with 400 Invalid host header
</summary>
```
Testing TrustedHostMiddleware with IPv6 addresses
==================================================

Test 1: Request to localhost
  Host header: localhost
  Status code: 200
  Response: Request accepted

Test 2: Request to IPv6 address with port
  Host header: [2001:db8::1]:8080
  Status code: 400
  Response: Invalid host header

Test 3: Request to IPv6 address without port
  Host header: [2001:db8::1]
  Status code: 400
  Response: Invalid host header

==================================================
Direct parsing demonstration:
  Original: [2001:db8::1]:8080
  After split(':')[0]: [2001
  Expected: [2001:db8::1] or 2001:db8::1
  Result: The parsed value '[2001' will not match any allowed_hosts
```
</details>

## Why This Is A Bug

The bug occurs in line 40 of `/home/npc/pbt/agentic-pbt/envs/fastapi_env/lib/python3.13/site-packages/starlette/middleware/trustedhost.py`:

```python
host = headers.get("host", "").split(":")[0]
```

This parsing logic incorrectly assumes that the first colon in a Host header always separates the hostname from the port. While this works for IPv4 addresses and domain names (e.g., "example.com:8080" → "example.com"), it fails for IPv6 addresses which inherently contain colons.

According to RFC 2732, IPv6 addresses in URLs and HTTP Host headers must be enclosed in square brackets to disambiguate them from port separators. The standard format is `[IPv6-address]:port`. When the middleware encounters a valid Host header like `[2001:db8::1]:8080`, it incorrectly splits on the first colon, producing `[2001` instead of the complete IPv6 address `[2001:db8::1]`.

This causes the middleware to reject all IPv6 requests, even when the IPv6 address is explicitly included in the `allowed_hosts` list. The middleware returns a 400 "Invalid host header" response for any IPv6 request, effectively breaking IPv6 support entirely.

## Relevant Context

FastAPI re-exports this middleware directly from Starlette without modification. The bug affects both frameworks equally. IPv6 adoption is increasing globally, with many cloud providers and modern infrastructure requiring IPv6 support. This bug makes it impossible to use TrustedHostMiddleware in IPv6-enabled environments.

The middleware's documentation mentions it validates "domain names" but doesn't explicitly exclude IP addresses. Since the middleware attempts to parse all Host headers and many production systems need to validate IPv6 addresses (especially in containerized or cloud environments), this limitation severely impacts real-world usage.

Starlette source code location: `/home/npc/pbt/agentic-pbt/envs/fastapi_env/lib/python3.13/site-packages/starlette/middleware/trustedhost.py:40`

## Proposed Fix

```diff
--- a/starlette/middleware/trustedhost.py
+++ b/starlette/middleware/trustedhost.py
@@ -37,7 +37,14 @@ class TrustedHostMiddleware:
             return

         headers = Headers(scope=scope)
-        host = headers.get("host", "").split(":")[0]
+        host_header = headers.get("host", "")
+
+        # Handle IPv6 addresses in bracket notation per RFC 2732
+        if host_header.startswith("["):
+            # IPv6: extract everything up to and including the closing bracket
+            host = host_header.split("]")[0] + "]" if "]" in host_header else host_header
+        else:
+            host = host_header.split(":")[0]
         is_valid_host = False
         found_www_redirect = False
         for pattern in self.allowed_hosts:
```