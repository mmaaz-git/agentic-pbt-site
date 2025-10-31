# Bug Report: TrustedHostMiddleware IPv6 Address Parsing Failure

**Target**: `starlette.middleware.trustedhost.TrustedHostMiddleware`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The TrustedHostMiddleware incorrectly parses IPv6 addresses in HTTP Host headers, causing all valid IPv6 requests to be rejected with a 400 error due to using `.split(":")[0]` which breaks on the colons within IPv6 addresses.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Property-based test using Hypothesis to find IPv6 parsing bugs in TrustedHostMiddleware.
This test generates various IPv6 addresses and verifies that the host extraction logic works correctly.
"""

from hypothesis import given, strategies as st, settings, Verbosity


def extract_host_current_implementation(host_header):
    """
    This mimics the current buggy behavior in starlette/middleware/trustedhost.py line 40:
    host = headers.get("host", "").split(":")[0]
    """
    return host_header.split(":")[0]


@st.composite
def ipv6_addresses(draw):
    """Generate valid IPv6 addresses in bracket notation."""
    segments = draw(st.lists(
        st.text(min_size=1, max_size=4, alphabet='0123456789abcdef'),
        min_size=2,
        max_size=8
    ))
    return "[" + ":".join(segments) + "]"


@given(ipv6_addresses())
@settings(max_examples=100, verbosity=Verbosity.verbose)
def test_ipv6_host_extraction_bug(ipv6_addr):
    """
    Test that IPv6 addresses are correctly extracted from Host headers.

    The Host header can contain:
    - IPv4: "example.com" or "example.com:8080"
    - IPv6: "[::1]" or "[::1]:8080"

    For IPv6, the address must be in brackets per RFC 3986.
    The host extraction should return the full bracketed IPv6 address.
    """
    extracted = extract_host_current_implementation(ipv6_addr)

    # The extracted host should be the full IPv6 address including brackets
    if extracted != ipv6_addr:
        raise AssertionError(
            f"IPv6 address parsing failed: "
            f"Input: '{ipv6_addr}' -> Extracted: '{extracted}' (Expected: '{ipv6_addr}')"
        )


if __name__ == "__main__":
    # Run the test to find a minimal failing example
    try:
        test_ipv6_host_extraction_bug()
    except Exception as e:
        print(f"Test failed as expected, demonstrating the bug: {e}")
```

<details>

<summary>
**Failing input**: `[0:0]`
</summary>
```
Running property-based test to find IPv6 parsing bugs...
============================================================
Trying example: test_ipv6_host_extraction_bug(
    ipv6_addr='[0:0]',
)
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/5/hypo.py", line 46, in test_ipv6_host_extraction_bug
    raise AssertionError(
    ...<2 lines>...
    )
AssertionError: IPv6 address parsing failed: Input: '[0:0]' -> Extracted: '[0' (Expected: '[0:0]')
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Minimal reproduction of the TrustedHostMiddleware IPv6 parsing bug.
This demonstrates how the current implementation incorrectly parses IPv6 addresses.
"""

def extract_host_current_implementation(host_header):
    """
    This mimics the current behavior in starlette/middleware/trustedhost.py line 40:
    host = headers.get("host", "").split(":")[0]
    """
    return host_header.split(":")[0]


def main():
    print("Testing IPv6 Host Header Parsing Bug")
    print("=" * 50)

    # Test cases: (input_host_header, expected_host, actual_result)
    test_cases = [
        # IPv6 localhost without port
        ("[::1]", "[::1]"),
        # IPv6 localhost with port
        ("[::1]:8000", "[::1]"),
        # Full IPv6 address without port
        ("[2001:db8::1]", "[2001:db8::1]"),
        # Full IPv6 address with port
        ("[2001:db8::1]:8080", "[2001:db8::1]"),
        # IPv6 with many segments
        ("[2001:0db8:85a3:0000:0000:8a2e:0370:7334]", "[2001:0db8:85a3:0000:0000:8a2e:0370:7334]"),
        # IPv6 with port and many segments
        ("[2001:0db8:85a3:0000:0000:8a2e:0370:7334]:443", "[2001:0db8:85a3:0000:0000:8a2e:0370:7334]"),
        # Minimal failing case from Hypothesis
        ("[0:0]", "[0:0]"),
    ]

    print("\nCurrent Implementation Results:")
    print("-" * 50)

    failures = []
    for host_header, expected_host in test_cases:
        actual = extract_host_current_implementation(host_header)
        status = "✓ PASS" if actual == expected_host else "✗ FAIL"
        print(f"Input:    '{host_header}'")
        print(f"Expected: '{expected_host}'")
        print(f"Actual:   '{actual}'")
        print(f"Status:   {status}")
        print()

        if actual != expected_host:
            failures.append((host_header, expected_host, actual))

    if failures:
        print("\nSUMMARY: Bug Confirmed!")
        print("-" * 50)
        print(f"Failed {len(failures)} out of {len(test_cases)} test cases.")
        print("\nThe bug occurs because split(':')[0] incorrectly assumes that")
        print("the first colon separates the host from the port, but IPv6")
        print("addresses contain multiple colons within the address itself.")
        print("\nFailed cases:")
        for host_header, expected, actual in failures:
            print(f"  '{host_header}' -> got '{actual}' instead of '{expected}'")

        print("\nImpact: Any application using TrustedHostMiddleware will reject")
        print("valid IPv6 requests, returning a 400 'Invalid host header' error.")
    else:
        print("\nAll tests passed (this should not happen with the buggy implementation)")

    return len(failures) > 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
```

<details>

<summary>
All test cases fail - 100% of IPv6 addresses are incorrectly parsed
</summary>
```
Testing IPv6 Host Header Parsing Bug
==================================================

Current Implementation Results:
--------------------------------------------------
Input:    '[::1]'
Expected: '[::1]'
Actual:   '['
Status:   ✗ FAIL

Input:    '[::1]:8000'
Expected: '[::1]'
Actual:   '['
Status:   ✗ FAIL

Input:    '[2001:db8::1]'
Expected: '[2001:db8::1]'
Actual:   '[2001'
Status:   ✗ FAIL

Input:    '[2001:db8::1]:8080'
Expected: '[2001:db8::1]'
Actual:   '[2001'
Status:   ✗ FAIL

Input:    '[2001:0db8:85a3:0000:0000:8a2e:0370:7334]'
Expected: '[2001:0db8:85a3:0000:0000:8a2e:0370:7334]'
Actual:   '[2001'
Status:   ✗ FAIL

Input:    '[2001:0db8:85a3:0000:0000:8a2e:0370:7334]:443'
Expected: '[2001:0db8:85a3:0000:0000:8a2e:0370:7334]'
Actual:   '[2001'
Status:   ✗ FAIL

Input:    '[0:0]'
Expected: '[0:0]'
Actual:   '[0'
Status:   ✗ FAIL


SUMMARY: Bug Confirmed!
--------------------------------------------------
Failed 7 out of 7 test cases.

The bug occurs because split(':')[0] incorrectly assumes that
the first colon separates the host from the port, but IPv6
addresses contain multiple colons within the address itself.

Failed cases:
  '[::1]' -> got '[' instead of '[::1]'
  '[::1]:8000' -> got '[' instead of '[::1]'
  '[2001:db8::1]' -> got '[2001' instead of '[2001:db8::1]'
  '[2001:db8::1]:8080' -> got '[2001' instead of '[2001:db8::1]'
  '[2001:0db8:85a3:0000:0000:8a2e:0370:7334]' -> got '[2001' instead of '[2001:0db8:85a3:0000:0000:8a2e:0370:7334]'
  '[2001:0db8:85a3:0000:0000:8a2e:0370:7334]:443' -> got '[2001' instead of '[2001:0db8:85a3:0000:0000:8a2e:0370:7334]'
  '[0:0]' -> got '[0' instead of '[0:0]'

Impact: Any application using TrustedHostMiddleware will reject
valid IPv6 requests, returning a 400 'Invalid host header' error.
```
</details>

## Why This Is A Bug

This violates RFC 3986 Section 3.2.2 which specifies that IPv6 addresses in URLs must be enclosed in square brackets. The HTTP Host header follows this same standard as specified in RFC 7230 Section 5.4. When a client sends an HTTP request to an IPv6 address, the Host header will contain the bracketed IPv6 address (e.g., `Host: [::1]` or `Host: [2001:db8::1]:8080`).

The bug in line 40 of `starlette/middleware/trustedhost.py`:
```python
host = headers.get("host", "").split(":")[0]
```

This implementation incorrectly assumes that the first colon in the Host header separates the hostname from the port number. However, IPv6 addresses contain multiple colons as part of the address notation itself. As a result:

1. **All IPv6 requests are rejected**: The middleware extracts an incorrect hostname (e.g., `[` or `[2001`) which doesn't match any allowed hosts, causing valid requests to fail with a 400 "Invalid host header" error.

2. **Security implications**: The extracted hostname doesn't match what was intended, potentially creating inconsistencies in security checks.

3. **No IPv6 support**: Applications using TrustedHostMiddleware cannot accept IPv6 connections, which is critical as IPv4 addresses become scarce and IPv6 adoption increases.

## Relevant Context

The TrustedHostMiddleware is a security middleware that validates the Host header against a list of allowed hosts to prevent host header injection attacks. It's commonly used in production deployments to ensure requests are only accepted for configured domains.

The current implementation at line 40 of `/home/npc/miniconda/lib/python3.13/site-packages/starlette/middleware/trustedhost.py` needs to handle both IPv4 and IPv6 address formats:
- IPv4/domain format: `example.com` or `example.com:8080`
- IPv6 format: `[2001:db8::1]` or `[2001:db8::1]:8080`

Starlette documentation: https://www.starlette.io/middleware/#trustedhost-middleware
RFC 3986 (URI Generic Syntax): https://tools.ietf.org/html/rfc3986#section-3.2.2
RFC 7230 (HTTP/1.1 Message Syntax): https://tools.ietf.org/html/rfc7230#section-5.4

## Proposed Fix

```diff
--- a/starlette/middleware/trustedhost.py
+++ b/starlette/middleware/trustedhost.py
@@ -37,7 +37,12 @@ class TrustedHostMiddleware:
             return

         headers = Headers(scope=scope)
-        host = headers.get("host", "").split(":")[0]
+        host_header = headers.get("host", "")
+        if host_header.startswith("["):
+            # IPv6 address - extract up to the closing bracket
+            host = host_header.split("]")[0] + "]" if "]" in host_header else host_header
+        else:
+            host = host_header.split(":")[0]
         is_valid_host = False
         found_www_redirect = False
         for pattern in self.allowed_hosts:
```