# Bug Report: TrustedHostMiddleware IPv6 Address Handling

**Target**: `starlette.middleware.trustedhost.TrustedHostMiddleware`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

TrustedHostMiddleware incorrectly extracts the hostname from IPv6 addresses in the Host header. It splits on the first colon, which breaks IPv6 addresses that contain multiple colons, making it impossible to allow IPv6 hosts.

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
```

**Failing input**: Any IPv6 address like `[::1]`, `[2001:db8::1]`, or `[::1]:8080`

## Reproducing the Bug

```python
from starlette.middleware.trustedhost import TrustedHostMiddleware

middleware = TrustedHostMiddleware(None, allowed_hosts=["[::1]"])

host_header = "[::1]"
extracted_host = host_header.split(":")[0]

print(f"Host header: {host_header}")
print(f"Extracted: {extracted_host}")
print(f"Expected: [::1]")

is_valid = False
for pattern in middleware.allowed_hosts:
    if extracted_host == pattern or (pattern.startswith("*") and extracted_host.endswith(pattern[1:])):
        is_valid = True
        break

print(f"Is valid: {is_valid}")

assert is_valid
```

## Why This Is A Bug

The code at line 40 in `trustedhost.py`:

```python
host = headers.get("host", "").split(":")[0]
```

This splits the Host header on the first colon to extract the hostname and remove the port. However, IPv6 addresses contain multiple colons, so this breaks:

- `[::1]` → `[` (should be `[::1]`)
- `[::1]:8080` → `[` (should be `[::1]`)
- `[2001:db8::1]` → `[2001` (should be `[2001:db8::1]`)
- `[2001:db8::1]:8000` → `[2001` (should be `[2001:db8::1]`)

IPv6 addresses in the Host header are enclosed in brackets per RFC 3986, and the port (if present) comes after the closing bracket. The current logic doesn't account for this.

## Fix

```diff
--- a/starlette/middleware/trustedhost.py
+++ b/starlette/middleware/trustedhost.py
@@ -37,7 +37,13 @@ class TrustedHostMiddleware:
             return

         headers = Headers(scope=scope)
-        host = headers.get("host", "").split(":")[0]
+        host_header = headers.get("host", "")
+
+        if host_header.startswith("["):
+            host = host_header.rsplit(":", 1)[0] if "]:" in host_header else host_header
+        else:
+            host = host_header.split(":", 1)[0]
+
         is_valid_host = False
         found_www_redirect = False
         for pattern in self.allowed_hosts:
```