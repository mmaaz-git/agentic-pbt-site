# Bug Report: TrustedHostMiddleware IPv6 Address Parsing Failure

**Target**: `starlette.middleware.trustedhost.TrustedHostMiddleware`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

TrustedHostMiddleware incorrectly parses IPv6 addresses when a port is specified in the Host header, causing it to reject valid IPv6 hosts even when they are explicitly allowed.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.testclient import TestClient


# Create a simple app
app = Starlette()

@app.route("/")
async def homepage(request):
    return PlainTextResponse("Hello")


@given(st.integers(min_value=1, max_value=65535))
def test_trustedhost_ipv6_port_handling(port):
    # Create middleware that allows IPv6 localhost
    middleware = TrustedHostMiddleware(
        app=app,
        allowed_hosts=["[::1]"]
    )

    # Create the host header as it would appear in an HTTP request
    host_header = f"[::1]:{port}"

    # This is what the middleware does (line 40 of trustedhost.py)
    extracted_host = host_header.split(":")[0]

    # The assertion that should pass but doesn't due to the bug
    assert extracted_host == "[::1]", (
        f"IPv6 host extraction failed for port {port}. "
        f"Host header: '{host_header}', "
        f"Expected: '[::1]', "
        f"Got: '{extracted_host}'"
    )


if __name__ == "__main__":
    # Run the test
    test_trustedhost_ipv6_port_handling()
```

<details>

<summary>
**Failing input**: `port=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/46/hypo.py", line 41, in <module>
    test_trustedhost_ipv6_port_handling()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/46/hypo.py", line 17, in test_trustedhost_ipv6_port_handling
    def test_trustedhost_ipv6_port_handling(port):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/46/hypo.py", line 31, in test_trustedhost_ipv6_port_handling
    assert extracted_host == "[::1]", (
           ^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: IPv6 host extraction failed for port 1. Host header: '[::1]:1', Expected: '[::1]', Got: '['
Falsifying example: test_trustedhost_ipv6_port_handling(
    port=1,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.testclient import TestClient

# Create a simple app
app = Starlette()

@app.route("/")
async def homepage(request):
    return PlainTextResponse("Hello")

# Test 1: Show the parsing issue directly
print("=== Direct Parsing Issue ===")
host_header = "[::1]:8000"
extracted_host = host_header.split(":")[0]
print(f"Host header: {host_header}")
print(f"Extracted host using split(':')[0]: {extracted_host}")
print(f"Expected: [::1]")
print(f"Bug: Got '{extracted_host}' instead of '[::1]'")
print()

# Test 2: Show that the middleware fails with IPv6
print("=== Middleware Behavior Test ===")
print("Testing with Host header: [::1]:8000")
print("Allowed hosts: ['[::1]']")
print()

# Wrap with TrustedHostMiddleware that should allow IPv6 localhost
middleware = TrustedHostMiddleware(app, allowed_hosts=["[::1]"])

# Create test client - use regular hostname to avoid TestClient IPv6 parsing issues
client = TestClient(middleware)

# Make a request with IPv6 host header
try:
    response = client.get("/", headers={"host": "[::1]:8000"})
    print(f"Response status code: {response.status_code}")
    print(f"Response text: {response.text}")
    print()

    if response.status_code == 400:
        print("BUG CONFIRMED: The middleware rejected a valid IPv6 host that was in allowed_hosts")
        print("This happens because the middleware extracts '[' instead of '[::1]' from the host header")
    else:
        print("Request succeeded (unexpected)")
except Exception as e:
    print(f"Error during request: {e}")
    print()

# Test 3: Demonstrate with other IPv6 addresses
print("=== Other IPv6 Examples ===")
test_cases = [
    "[2001:db8::1]:3000",
    "[fe80::1]:80",
    "[::ffff:192.0.2.1]:443"
]

for test_host in test_cases:
    extracted = test_host.split(":")[0]
    print(f"Host: {test_host} -> Extracted: '{extracted}' (should be '{test_host.rsplit(":", 1)[0]}')")
```

<details>

<summary>
Response status code: 400 - Invalid host header
</summary>
```
=== Direct Parsing Issue ===
Host header: [::1]:8000
Extracted host using split(':')[0]: [
Expected: [::1]
Bug: Got '[' instead of '[::1]'

=== Middleware Behavior Test ===
Testing with Host header: [::1]:8000
Allowed hosts: ['[::1]']

Response status code: 400
Response text: Invalid host header

BUG CONFIRMED: The middleware rejected a valid IPv6 host that was in allowed_hosts
This happens because the middleware extracts '[' instead of '[::1]' from the host header
=== Other IPv6 Examples ===
Host: [2001:db8::1]:3000 -> Extracted: '[2001' (should be '[2001:db8::1]')
Host: [fe80::1]:80 -> Extracted: '[fe80' (should be '[fe80::1]')
Host: [::ffff:192.0.2.1]:443 -> Extracted: '[' (should be '[::ffff:192.0.2.1]')
```
</details>

## Why This Is A Bug

The middleware violates HTTP/URL standards for IPv6 address representation. According to RFC 3986 and RFC 2732, IPv6 addresses in URLs and HTTP Host headers must be enclosed in square brackets (e.g., `[::1]` or `[2001:db8::1]`). When a port is included, the format is `[IPv6]:port`.

The bug occurs at line 40 of `trustedhost.py`:
```python
host = headers.get("host", "").split(":")[0]
```

This naive string split breaks on the first colon, which is incorrect for IPv6 addresses that contain multiple colons. For example:
- `[::1]:8000` splits to `[` instead of `[::1]`
- `[2001:db8::1]:3000` splits to `[2001` instead of `[2001:db8::1]`

The consequence is that **any IPv6 address with a port will always be rejected** by the middleware, even when the IPv6 address is explicitly included in `allowed_hosts`. This breaks legitimate deployments in IPv6 environments and prevents proper host validation for IPv6 traffic.

## Relevant Context

The TrustedHostMiddleware is designed to validate HTTP Host headers against an allowed list to prevent host header injection attacks. While the documentation mentions "domain names," the middleware already handles IPv4 addresses correctly (e.g., `192.168.1.1:8000` works properly). The failure to handle IPv6 addresses appears to be an oversight in the parsing logic rather than an intentional limitation.

IPv6 support is essential for modern web applications as:
- IPv6 adoption is increasing globally due to IPv4 address exhaustion
- Many cloud providers and CDNs now prefer or require IPv6
- Development and testing environments often use IPv6 localhost (`[::1]`)
- The fix is straightforward and doesn't break existing functionality

Documentation: https://www.starlette.io/middleware/#trustedhostmiddleware
RFC 3986 (URI Generic Syntax): https://www.rfc-editor.org/rfc/rfc3986.html#section-3.2.2
RFC 2732 (IPv6 Literal Addresses in URLs): https://www.rfc-editor.org/rfc/rfc2732.html

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
+        # Handle IPv6 addresses which are enclosed in brackets like [::1]:8000
+        if host_header.startswith("[") and "]" in host_header:
+            # Extract the IPv6 address including brackets
+            host = host_header.split("]")[0] + "]"
+        else:
+            host = host_header.split(":")[0]
         is_valid_host = False
         found_www_redirect = False
         for pattern in self.allowed_hosts:
```