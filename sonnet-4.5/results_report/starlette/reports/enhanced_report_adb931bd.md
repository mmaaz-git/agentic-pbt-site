# Bug Report: starlette.middleware.trustedhost.TrustedHostMiddleware IPv6 Address Parsing Failure

**Target**: `starlette.middleware.trustedhost.TrustedHostMiddleware`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

TrustedHostMiddleware fails to parse IPv6 addresses when a port is present in the Host header, causing all valid IPv6 requests with ports to be rejected with HTTP 400 "Invalid host header" errors.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Property-based test for TrustedHostMiddleware IPv6 parsing bug using Hypothesis."""

from hypothesis import given, strategies as st, example


@example(ipv6="::1", port="8000")
@example(ipv6="2001:db8::1", port="8080")
@given(
    ipv6=st.text(min_size=1, max_size=20),
    port=st.text(min_size=1, max_size=5, alphabet=st.characters(min_codepoint=48, max_codepoint=57))
)
def test_trustedhost_ipv6_parsing(ipv6, port):
    """Test that IPv6 addresses with ports are correctly parsed."""
    host_with_port = f"[{ipv6}]:{port}"

    # This is how the current middleware extracts the host
    host_extracted = host_with_port.split(":")[0]

    # For IPv6 addresses in brackets, we should extract the address
    if "[" in host_with_port:
        bracket_close = host_with_port.index("]")
        correct_host = host_with_port[1:bracket_close]

        # The extracted host should either be the bracketed form or just the IPv6 address
        assert host_extracted == f"[{ipv6}]" or host_extracted == ipv6, (
            f"IPv6 address should be correctly extracted. "
            f"Host header: {host_with_port}, "
            f"Extracted: '{host_extracted}', "
            f"Expected: '[{ipv6}]' or '{ipv6}'"
        )


if __name__ == "__main__":
    # Run the test
    test_trustedhost_ipv6_parsing()
```

<details>

<summary>
**Failing input**: `ipv6='::1', port='8000'` and `ipv6='2001:db8::1', port='8080'`
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/40/hypo.py", line 36, in <module>
  |     test_trustedhost_ipv6_parsing()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/40/hypo.py", line 8, in test_trustedhost_ipv6_parsing
  |     @example(ipv6="2001:db8::1", port="8080")
  |                    ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
  |     _raise_to_user(errors, state.settings, [], " in explicit examples")
  |     ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures in explicit examples. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/40/hypo.py", line 26, in test_trustedhost_ipv6_parsing
    |     assert host_extracted == f"[{ipv6}]" or host_extracted == ipv6, (
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    | AssertionError: IPv6 address should be correctly extracted. Host header: [::1]:8000, Extracted: '[', Expected: '[::1]' or '::1'
    | Falsifying explicit example: test_trustedhost_ipv6_parsing(
    |     ipv6='::1',
    |     port='8000',
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/40/hypo.py", line 26, in test_trustedhost_ipv6_parsing
    |     assert host_extracted == f"[{ipv6}]" or host_extracted == ipv6, (
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    | AssertionError: IPv6 address should be correctly extracted. Host header: [2001:db8::1]:8080, Extracted: '[2001', Expected: '[2001:db8::1]' or '2001:db8::1'
    | Falsifying explicit example: test_trustedhost_ipv6_parsing(
    |     ipv6='2001:db8::1',
    |     port='8080',
    | )
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction of the TrustedHostMiddleware IPv6 parsing bug."""

from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.datastructures import Headers


async def dummy_app(scope, receive, send):
    """Dummy ASGI application for testing."""
    pass


# Create middleware with IPv6 address in allowed hosts
middleware = TrustedHostMiddleware(
    app=dummy_app,
    allowed_hosts=["::1", "2001:db8::1"],
)

# Test case 1: IPv6 address with port in Host header
print("=== Test Case 1: IPv6 [::1]:8000 ===")
scope = {
    "type": "http",
    "headers": [[b"host", b"[::1]:8000"]],
}
headers = Headers(scope=scope)
host_extracted = headers.get("host", "").split(":")[0]
print(f"Host header: [::1]:8000")
print(f"Extracted by middleware logic: '{host_extracted}'")
print(f"Expected: '::1' or '[::1]'")
print(f"Result: {'✗ FAIL' if host_extracted == '[' else '✓ PASS'}")
print()

# Test case 2: Another IPv6 address with port
print("=== Test Case 2: IPv6 [2001:db8::1]:8080 ===")
scope2 = {
    "type": "http",
    "headers": [[b"host", b"[2001:db8::1]:8080"]],
}
headers2 = Headers(scope=scope2)
host_extracted2 = headers2.get("host", "").split(":")[0]
print(f"Host header: [2001:db8::1]:8080")
print(f"Extracted by middleware logic: '{host_extracted2}'")
print(f"Expected: '2001:db8::1' or '[2001:db8::1]'")
print(f"Result: {'✗ FAIL' if host_extracted2 == '[2001' else '✓ PASS'}")
print()

# Test case 3: IPv4 address with port (should work correctly)
print("=== Test Case 3: IPv4 192.168.1.1:8000 ===")
scope3 = {
    "type": "http",
    "headers": [[b"host", b"192.168.1.1:8000"]],
}
headers3 = Headers(scope=scope3)
host_extracted3 = headers3.get("host", "").split(":")[0]
print(f"Host header: 192.168.1.1:8000")
print(f"Extracted by middleware logic: '{host_extracted3}'")
print(f"Expected: '192.168.1.1'")
print(f"Result: {'✓ PASS' if host_extracted3 == '192.168.1.1' else '✗ FAIL'}")
print()

# Demonstrate the actual impact on middleware validation
print("=== Impact on Middleware Validation ===")
print("Testing if valid IPv6 hosts are accepted by the middleware...")
print()

# Simulate what happens internally in the middleware
for test_host in ["[::1]:8000", "[2001:db8::1]:8080"]:
    host_header = test_host
    extracted_host = host_header.split(":")[0]

    # Check against allowed hosts
    is_valid = False
    for allowed in ["::1", "2001:db8::1"]:
        if extracted_host == allowed:
            is_valid = True
            break

    print(f"Host header: {host_header}")
    print(f"  → Extracted: '{extracted_host}'")
    print(f"  → Valid host? {is_valid}")
    print(f"  → Result: Request would be {'accepted ✓' if is_valid else 'rejected with 400 Invalid host header ✗'}")
    print()
```

<details>

<summary>
All IPv6 requests with ports are incorrectly rejected with HTTP 400 errors
</summary>
```
=== Test Case 1: IPv6 [::1]:8000 ===
Host header: [::1]:8000
Extracted by middleware logic: '['
Expected: '::1' or '[::1]'
Result: ✗ FAIL

=== Test Case 2: IPv6 [2001:db8::1]:8080 ===
Host header: [2001:db8::1]:8080
Extracted by middleware logic: '[2001'
Expected: '2001:db8::1' or '[2001:db8::1]'
Result: ✗ FAIL

=== Test Case 3: IPv4 192.168.1.1:8000 ===
Host header: 192.168.1.1:8000
Extracted by middleware logic: '192.168.1.1'
Expected: '192.168.1.1'
Result: ✓ PASS

=== Impact on Middleware Validation ===
Testing if valid IPv6 hosts are accepted by the middleware...

Host header: [::1]:8000
  → Extracted: '['
  → Valid host? False
  → Result: Request would be rejected with 400 Invalid host header ✗

Host header: [2001:db8::1]:8080
  → Extracted: '[2001'
  → Valid host? False
  → Result: Request would be rejected with 400 Invalid host header ✗
```
</details>

## Why This Is A Bug

The middleware violates RFC 3986 and RFC 2732, which explicitly define that IPv6 literal addresses in URLs and HTTP headers must be enclosed in square brackets when accompanied by a port number (e.g., `[2001:db8::1]:8080`). The bug occurs at line 40 of `/home/npc/pbt/agentic-pbt/envs/starlette_env/lib/python3.13/site-packages/starlette/middleware/trustedhost.py`:

```python
host = headers.get("host", "").split(":")[0]
```

This naive string split on the first colon character fails catastrophically for IPv6 addresses because:
1. IPv6 addresses contain multiple colons as part of the address notation (e.g., `2001:db8::1`)
2. When wrapped in brackets with a port `[2001:db8::1]:8080`, the split extracts `[2001` instead of `2001:db8::1`
3. For localhost IPv6 `[::1]:8000`, it extracts just `[` instead of `::1`

This makes the middleware completely unusable for:
- IPv6-only networks (increasingly common with IPv4 exhaustion)
- Dual-stack deployments where clients connect via IPv6
- Local development using IPv6 localhost (`::1`)
- Cloud environments that prefer IPv6 connectivity

The middleware correctly handles IPv4 addresses and domain names but completely fails on a fundamental internet protocol standard.

## Relevant Context

**RFC 3986 Section 3.2.2** states:
> "A host identified by an Internet Protocol literal address, version 6 [RFC3513] or later, is distinguished by enclosing the IP literal within square brackets ('[' and ']'). This is the only place where square bracket characters are allowed in the URI syntax."

**RFC 2732** specifically addresses this format:
> "To use a literal IPv6 address in a URL, the literal address should be enclosed in '[' and ']' characters."

Example valid Host headers per the RFCs:
- `[::1]:8000` - IPv6 localhost with port 8000
- `[2001:db8::1]:443` - IPv6 address with HTTPS port
- `[fe80::1%eth0]:3000` - Link-local IPv6 with zone ID and port

The Starlette documentation doesn't explicitly mention IPv6 support, but as a modern ASGI framework handling HTTP headers, supporting standard-compliant Host headers is a reasonable expectation. The middleware already attempts to handle IP addresses (IPv4 works correctly), making the IPv6 failure a clear bug rather than an unsupported feature.

## Proposed Fix

```diff
--- a/starlette/middleware/trustedhost.py
+++ b/starlette/middleware/trustedhost.py
@@ -37,7 +37,17 @@ class TrustedHostMiddleware:
             return

         headers = Headers(scope=scope)
-        host = headers.get("host", "").split(":")[0]
+        host_header = headers.get("host", "")
+
+        # Handle IPv6 addresses in brackets [IPv6]:port per RFC 3986
+        if host_header.startswith("["):
+            # Find the closing bracket
+            bracket_end = host_header.find("]")
+            if bracket_end != -1:
+                host = host_header[1:bracket_end]
+            else:
+                host = host_header  # Malformed, let validation fail
+        else:
+            host = host_header.split(":")[0]
         is_valid_host = False
         found_www_redirect = False
         for pattern in self.allowed_hosts:
```