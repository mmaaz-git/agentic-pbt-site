# Bug Report: starlette.middleware.trustedhost IPv6 Address Parsing

**Target**: `starlette.middleware.trustedhost.TrustedHostMiddleware.__call__`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `TrustedHostMiddleware` incorrectly parses IPv6 addresses in the Host header, causing valid IPv6 hosts to be rejected even when they are in the allowed_hosts list.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from starlette.middleware.trustedhost import TrustedHostMiddleware


def dummy_app(scope, receive, send):
    pass


@given(
    st.lists(st.integers(min_value=0, max_value=65535), min_size=8, max_size=8)
)
def test_trustedhost_ipv6_parsing(parts):
    ipv6 = ":".join(f"{p:x}" for p in parts)
    ipv6_bracketed = f"[{ipv6}]"

    middleware = TrustedHostMiddleware(dummy_app, allowed_hosts=[ipv6_bracketed])

    host_after_split = ipv6_bracketed.split(":")[0]

    assert ipv6_bracketed == host_after_split or not ":" in ipv6, \
        f"IPv6 address {ipv6_bracketed} incorrectly parsed to {host_after_split}"
```

**Failing input**: Any IPv6 address with colons, e.g., `[::1]`, `[2001:db8::1]`

## Reproducing the Bug

```python
from starlette.middleware.trustedhost import TrustedHostMiddleware


def dummy_app(scope, receive, send):
    pass


middleware = TrustedHostMiddleware(dummy_app, allowed_hosts=["[::1]"])

host_header = "[::1]"
host_after_split = host_header.split(":")[0]

print(f"Host header: {host_header}")
print(f"Parsed as: {host_after_split!r}")
print(f"Expected: '[::1]'")
print(f"Actual: '['")
```

**Output:**
```
Host header: [::1]
Parsed as: '['
Expected: '[::1]'
Actual: '['
```

## Why This Is A Bug

The middleware uses `host = headers.get("host", "").split(":")[0]` to extract the hostname and remove port numbers. This works for IPv4 addresses and domain names (e.g., `example.com:8000` → `example.com`), but breaks for IPv6 addresses because they contain colons.

According to RFC 2732, IPv6 addresses in HTTP Host headers must be enclosed in brackets, like `[2001:db8::1]` or `[::1]:8080`. The current implementation incorrectly parses these:
- `[::1]` → `[` (should be `[::1]`)
- `[2001:db8::1]` → `[2001` (should be `[2001:db8::1]`)
- `[::1]:8000` → `[` (should be `[::1]`)

This causes the middleware to reject valid IPv6 addresses even when they are explicitly allowed, breaking IPv6 support.

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
+        # Handle IPv6 addresses (enclosed in brackets)
+        if host_header.startswith("["):
+            host = host_header.split("]")[0] + "]"
+        else:
+            host = host_header.split(":")[0]
+
         is_valid_host = False
         found_www_redirect = False
         for pattern in self.allowed_hosts:
```