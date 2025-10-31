# Bug Report: TrustedHostMiddleware IPv6 Address Parsing Failure

**Target**: `starlette.middleware.trustedhost.TrustedHostMiddleware`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

TrustedHostMiddleware fails to correctly parse IPv6 addresses in Host headers due to using `.split(":")` which breaks on the first colon, making it impossible to validate or allow any IPv6 hosts.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
from starlette.middleware.trustedhost import TrustedHostMiddleware

@given(st.sampled_from(["[::1]", "[2001:db8::1]", "[fe80::1]"]))
@settings(max_examples=50)
def test_trustedhost_ipv6_addresses(ipv6_address):
    middleware = TrustedHostMiddleware(None, allowed_hosts=[ipv6_address])

    extracted = ipv6_address.split(":")[0]

    is_valid = False
    for pattern in middleware.allowed_hosts:
        if extracted == pattern or (pattern.startswith("*") and extracted.endswith(pattern[1:])):
            is_valid = True
            break

    assert is_valid is True

if __name__ == "__main__":
    test_trustedhost_ipv6_addresses()
```

<details>

<summary>
**Failing input**: `[::1]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/49/hypo.py", line 20, in <module>
    test_trustedhost_ipv6_addresses()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/49/hypo.py", line 5, in test_trustedhost_ipv6_addresses
    @settings(max_examples=50)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/49/hypo.py", line 17, in test_trustedhost_ipv6_addresses
    assert is_valid is True
           ^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_trustedhost_ipv6_addresses(
    ipv6_address='[::1]',
)
Exit code: 1
```
</details>

## Reproducing the Bug

```python
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.testclient import TestClient

# Create a simple app with TrustedHostMiddleware
app = Starlette()

# Configure middleware to accept IPv6 loopback address
middleware = TrustedHostMiddleware(app, allowed_hosts=["[::1]"])

@app.route("/")
async def homepage(request):
    return JSONResponse({"message": "Hello, world!"})

# Apply the middleware
app = TrustedHostMiddleware(app, allowed_hosts=["[::1]"])

# Test with a TestClient using IPv6 address
client = TestClient(app)

print("Testing IPv6 host header: [::1]")
print("=" * 50)

# First demonstrate the internal parsing issue
host_header = "[::1]"
extracted_host = host_header.split(":")[0]
print(f"Host header value: {host_header}")
print(f"What split(':')[0] extracts: {extracted_host}")
print(f"Expected extraction: [::1]")
print(f"Does '{extracted_host}' match '[::1]'? {extracted_host == '[::1]'}")
print()

# Now test the actual middleware behavior
print("Testing actual middleware behavior:")
print("-" * 50)

try:
    # This should work if IPv6 is properly supported, but it will fail
    response = client.get("/", headers={"host": "[::1]"})
    print(f"Response status code: {response.status_code}")
    print(f"Response content: {response.text}")
except Exception as e:
    print(f"Request failed with exception: {e}")

print()
print("Testing with IPv6 address with port: [::1]:8080")
print("-" * 50)
host_with_port = "[::1]:8080"
extracted_with_port = host_with_port.split(":")[0]
print(f"Host header value: {host_with_port}")
print(f"What split(':')[0] extracts: {extracted_with_port}")
print(f"Expected extraction: [::1]")

print()
print("Testing with full IPv6 address: [2001:db8::1]")
print("-" * 50)
full_ipv6 = "[2001:db8::1]"
extracted_full = full_ipv6.split(":")[0]
print(f"Host header value: {full_ipv6}")
print(f"What split(':')[0] extracts: {extracted_full}")
print(f"Expected extraction: [2001:db8::1]")

print()
print("For comparison, testing IPv4 with port (works correctly):")
print("-" * 50)
ipv4_with_port = "127.0.0.1:8080"
extracted_ipv4 = ipv4_with_port.split(":")[0]
print(f"Host header value: {ipv4_with_port}")
print(f"What split(':')[0] extracts: {extracted_ipv4}")
print(f"Expected extraction: 127.0.0.1")
print(f"Correctly extracted? {extracted_ipv4 == '127.0.0.1'}")
```

<details>

<summary>
Middleware returns 400 "Invalid host header" for valid IPv6 address
</summary>
```
Testing IPv6 host header: [::1]
==================================================
Host header value: [::1]
What split(':')[0] extracts: [
Expected extraction: [::1]
Does '[' match '[::1]'? False

Testing actual middleware behavior:
--------------------------------------------------
Response status code: 400
Response content: Invalid host header

Testing with IPv6 address with port: [::1]:8080
--------------------------------------------------
Host header value: [::1]:8080
What split(':')[0] extracts: [
Expected extraction: [::1]

Testing with full IPv6 address: [2001:db8::1]
--------------------------------------------------
Host header value: [2001:db8::1]
What split(':')[0] extracts: [2001
Expected extraction: [2001:db8::1]

For comparison, testing IPv4 with port (works correctly):
--------------------------------------------------
Host header value: 127.0.0.1:8080
What split(':')[0] extracts: 127.0.0.1
Expected extraction: 127.0.0.1
Correctly extracted? True
Exit code: 0
```
</details>

## Why This Is A Bug

The bug occurs at line 40 of `/lib/python3.13/site-packages/starlette/middleware/trustedhost.py`:

```python
host = headers.get("host", "").split(":")[0]
```

This line attempts to remove port numbers from the Host header by splitting on the first colon. However, this fundamentally breaks IPv6 addresses because:

1. **IPv6 addresses contain colons as separators**: IPv6 addresses like `::1` or `2001:db8::1` use colons as part of their notation.

2. **RFC 3986 compliance violation**: According to RFC 3986 Section 3.2.2, IPv6 addresses in URLs and Host headers MUST be enclosed in square brackets. The format is `[IPv6address]` or `[IPv6address]:port`. The current implementation violates this standard.

3. **Incorrect extraction results**:
   - `[::1]` → extracts `[` (should be `[::1]`)
   - `[2001:db8::1]` → extracts `[2001` (should be `[2001:db8::1]`)
   - `[::1]:8080` → extracts `[` (should be `[::1]`)

4. **No workaround possible**: Users cannot configure the middleware to accept IPv6 addresses. Even if they add `[::1]` to `allowed_hosts`, the middleware will always extract `[` and fail to match.

5. **Security implications**: This bug forces users to either disable host validation entirely (`allowed_hosts=["*"]`) or not use the middleware at all when IPv6 is required, both of which compromise security.

## Relevant Context

The TrustedHostMiddleware is designed to validate that incoming requests have a Host header matching an allowed list of hosts. This is a security feature to prevent Host header injection attacks. However, the current implementation makes it unusable with IPv6, which is increasingly common in:

- Cloud deployments (AWS, GCP, Azure all support IPv6)
- Container orchestration (Kubernetes supports IPv6)
- Modern networks (many ISPs now provide IPv6)
- Local development (localhost can resolve to `::1`)

The middleware correctly handles:
- IPv4 addresses: `127.0.0.1`
- IPv4 with ports: `127.0.0.1:8080`
- Hostnames: `example.com`
- Hostnames with ports: `example.com:8080`
- Wildcard patterns: `*.example.com`

But fails on ALL IPv6 addresses due to the parsing logic.

Relevant documentation:
- [RFC 3986 Section 3.2.2](https://datatracker.ietf.org/doc/html/rfc3986#section-3.2.2) - IPv6 address format in URIs
- [Starlette middleware documentation](https://www.starlette.io/middleware/#trustedhostmiddleware)
- Source code: `starlette/middleware/trustedhost.py` line 40

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
+        # IPv6 addresses are enclosed in brackets: [addr] or [addr]:port
+        if host_header.startswith("["):
+            # Find the closing bracket and extract everything up to it
+            bracket_end = host_header.find("]")
+            host = host_header[:bracket_end + 1] if bracket_end != -1 else host_header
+        else:
+            # IPv4 or hostname: split on first colon to remove port
+            host = host_header.split(":", 1)[0]
         is_valid_host = False
         found_www_redirect = False
         for pattern in self.allowed_hosts:
```