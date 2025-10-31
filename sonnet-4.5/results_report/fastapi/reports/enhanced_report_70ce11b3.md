# Bug Report: TrustedHostMiddleware IPv6 Address Parsing Failure

**Target**: `starlette.middleware.trustedhost.TrustedHostMiddleware`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The TrustedHostMiddleware fails to correctly extract IPv6 addresses from the Host header due to using `.split(":")[0]` which truncates at the first colon, causing all valid IPv6 requests to be rejected with 400 errors.

## Property-Based Test

```python
#!/usr/bin/env python3
from hypothesis import given, strategies as st
import pytest


def extract_host_current_implementation(host_header):
    """This is the exact implementation from line 40 of trustedhost.py"""
    return host_header.split(":")[0]


@st.composite
def ipv6_addresses(draw):
    segments = draw(st.lists(
        st.text(min_size=1, max_size=4, alphabet='0123456789abcdef'),
        min_size=2,
        max_size=8
    ))
    return "[" + ":".join(segments) + "]"


@given(ipv6_addresses())
def test_ipv6_host_extraction_bug(ipv6_addr):
    extracted = extract_host_current_implementation(ipv6_addr)

    if extracted != ipv6_addr:
        pytest.fail(
            f"IPv6 address parsing failed: "
            f"Input: '{ipv6_addr}' -> Extracted: '{extracted}' (Expected: '{ipv6_addr}')"
        )


if __name__ == "__main__":
    test_ipv6_host_extraction_bug()
```

<details>

<summary>
**Failing input**: `[0:0]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/42/hypo.py", line 33, in <module>
    test_ipv6_host_extraction_bug()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/42/hypo.py", line 22, in test_ipv6_host_extraction_bug
    def test_ipv6_host_extraction_bug(ipv6_addr):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/42/hypo.py", line 26, in test_ipv6_host_extraction_bug
    pytest.fail(
    ~~~~~~~~~~~^
        f"IPv6 address parsing failed: "
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        f"Input: '{ipv6_addr}' -> Extracted: '{extracted}' (Expected: '{ipv6_addr}')"
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/home/npc/miniconda/lib/python3.13/site-packages/_pytest/outcomes.py", line 177, in fail
    raise Failed(msg=reason, pytrace=pytrace)
Failed: IPv6 address parsing failed: Input: '[0:0]' -> Extracted: '[0' (Expected: '[0:0]')
Falsifying example: test_ipv6_host_extraction_bug(
    ipv6_addr='[0:0]',  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction of IPv6 address parsing bug in TrustedHostMiddleware"""

def extract_host_current_implementation(host_header):
    """This is the exact implementation from line 40 of trustedhost.py"""
    return host_header.split(":")[0]

# Test cases showing the bug
test_cases = [
    "[::1]",
    "[::1]:8000",
    "[2001:db8::1]",
    "[2001:db8::1]:8080",
    "[fe80::1%eth0]",
    "[::ffff:192.0.2.1]",
    "[0:0]",
    "[::]",
]

print("IPv6 Address Parsing Bug in TrustedHostMiddleware")
print("=" * 50)
print()

for host_header in test_cases:
    extracted = extract_host_current_implementation(host_header)
    expected = host_header.split("]")[0] + "]" if host_header.startswith("[") else host_header.split(":")[0]

    print(f"Input:    '{host_header}'")
    print(f"Output:   '{extracted}'")
    print(f"Expected: '{expected}'")
    print(f"FAIL: {extracted != expected}")
    print()
```

<details>

<summary>
All IPv6 addresses fail parsing - extraction stops at first colon
</summary>
```
IPv6 Address Parsing Bug in TrustedHostMiddleware
==================================================

Input:    '[::1]'
Output:   '['
Expected: '[::1]'
FAIL: True

Input:    '[::1]:8000'
Output:   '['
Expected: '[::1]'
FAIL: True

Input:    '[2001:db8::1]'
Output:   '[2001'
Expected: '[2001:db8::1]'
FAIL: True

Input:    '[2001:db8::1]:8080'
Output:   '[2001'
Expected: '[2001:db8::1]'
FAIL: True

Input:    '[fe80::1%eth0]'
Output:   '[fe80'
Expected: '[fe80::1%eth0]'
FAIL: True

Input:    '[::ffff:192.0.2.1]'
Output:   '['
Expected: '[::ffff:192.0.2.1]'
FAIL: True

Input:    '[0:0]'
Output:   '[0'
Expected: '[0:0]'
FAIL: True

Input:    '[::]'
Output:   '['
Expected: '[::]'
FAIL: True

```
</details>

## Why This Is A Bug

This violates RFC 3986, which mandates that IPv6 addresses in URIs must be enclosed in square brackets. The HTTP Host header follows this standard, meaning any client connecting to an IPv6 address will send headers like `Host: [::1]` or `Host: [2001:db8::1]:8080`.

The bug occurs at line 40 of `/home/npc/pbt/agentic-pbt/envs/fastapi_env/lib/python3.13/site-packages/starlette/middleware/trustedhost.py`:
```python
host = headers.get("host", "").split(":")[0]
```

This implementation assumes only a single colon exists (for port separation), but IPv6 addresses contain multiple colons. The extraction fails catastrophically:
- Every IPv6 address gets truncated at the first colon
- The extracted "host" becomes just `[` or `[xxxx` (partial address)
- This will never match any valid entry in `allowed_hosts`
- All legitimate IPv6 requests get rejected with 400 "Invalid host header"

This is not an edge case - it's a complete inability to handle standard IPv6 addresses, violating HTTP specifications and making the middleware unusable for any service binding to IPv6 addresses.

## Relevant Context

The TrustedHostMiddleware is designed to prevent Host header injection attacks by validating that incoming requests have a Host header matching a configured allowlist. This is critical security middleware.

Key documentation and specifications:
- RFC 3986 Section 3.2.2 defines IPv6 literal format in URIs: https://www.rfc-editor.org/rfc/rfc3986#section-3.2.2
- Starlette TrustedHostMiddleware source: https://github.com/encode/starlette/blob/master/starlette/middleware/trustedhost.py

The middleware works correctly for:
- Domain names: `example.com`, `*.example.com`
- IPv4 addresses: `192.168.1.1`, `192.168.1.1:8080`

But fails for ALL IPv6 addresses:
- Localhost: `[::1]`
- Link-local: `[fe80::1%eth0]`
- Global unicast: `[2001:db8::1]`
- IPv4-mapped: `[::ffff:192.0.2.1]`

With increasing IPv6 adoption and IPv4 exhaustion, this bug prevents the security middleware from working with modern network configurations.

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
+            # IPv6 address - extract up to closing bracket
+            host = host_header.split("]")[0] + "]"
+        else:
+            host = host_header.split(":")[0]
         is_valid_host = False
         found_www_redirect = False
         for pattern in self.allowed_hosts:
```