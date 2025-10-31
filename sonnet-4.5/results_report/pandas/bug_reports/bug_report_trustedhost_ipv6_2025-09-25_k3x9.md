# Bug Report: TrustedHostMiddleware IPv6 Address Parsing

**Target**: `starlette.middleware.trustedhost.TrustedHostMiddleware`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `TrustedHostMiddleware` incorrectly parses IPv6 addresses in the Host header, causing valid IPv6 requests to be rejected. The bug occurs because the code uses `.split(":")[0]` to extract the hostname, which breaks IPv6 addresses that contain multiple colons.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pytest


def extract_host_current_implementation(host_header):
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
```

**Failing input**: `[0:0]` (Hypothesis found this as the minimal failing example)

## Reproducing the Bug

```python
test_cases = [
    ("[::1]", "["),
    ("[::1]:8000", "["),
    ("[2001:db8::1]", "[2001"),
    ("[2001:db8::1]:8080", "[2001"),
]

for host_header, actual_output in test_cases:
    extracted = host_header.split(":")[0]
    print(f"Input: '{host_header}' -> Output: '{extracted}'")
```

Expected behavior: IPv6 addresses should be extracted correctly:
- `[::1]` → `[::1]`
- `[::1]:8000` → `[::1]`
- `[2001:db8::1]` → `[2001:db8::1]`
- `[2001:db8::1]:8080` → `[2001:db8::1]`

Actual behavior:
- `[::1]` → `[`
- `[::1]:8000` → `[`
- `[2001:db8::1]` → `[2001`
- `[2001:db8::1]:8080` → `[2001`

## Why This Is A Bug

According to RFC 3986, IPv6 addresses in URLs must be enclosed in square brackets. The HTTP Host header follows this standard. When a client sends a request to an IPv6 address, the Host header will contain the bracketed address (e.g., `[::1]` or `[::1]:8000`).

The current implementation in line 40 of `trustedhost.py`:
```python
host = headers.get("host", "").split(":")[0]
```

This incorrectly assumes that only one colon exists (for separating host from port), but IPv6 addresses contain multiple colons. This causes:
1. Valid IPv6 requests to be rejected with a 400 error
2. Security issues if the extracted hostname doesn't match what was intended

## Fix

```diff
--- a/starlette/middleware/trustedhost.py
+++ b/starlette/middleware/trustedhost.py
@@ -37,7 +37,12 @@ class TrustedHostMiddleware:
             return

         headers = Headers(scope=scope)
-        host = headers.get("host", "").split(":")[0]
+        host_header = headers.get("host", "")
+        if host_header.startswith("["):
+            # IPv6 address
+            host = host_header.split("]")[0] + "]"
+        else:
+            host = host_header.split(":")[0]
         is_valid_host = False
         found_www_redirect = False
         for pattern in self.allowed_hosts:
```

This fix correctly handles both IPv4 and IPv6 addresses by detecting the opening bracket and extracting up to the closing bracket for IPv6 addresses.