# Bug Report: starlette.middleware.trustedhost.TrustedHostMiddleware IPv6 Address Parsing

**Target**: `starlette.middleware.trustedhost.TrustedHostMiddleware`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`TrustedHostMiddleware` incorrectly parses IPv6 addresses when a port is present in the `Host` header. The middleware splits on `:` to extract the hostname from `host:port`, but IPv6 addresses contain colons (e.g., `[::1]:8000`), causing the extraction to fail.

## Property-Based Test

```python
from hypothesis import given, strategies as st, example


@example(ipv6="::1", port="8000")
@example(ipv6="2001:db8::1", port="8080")
@given(
    st.text(min_size=1, max_size=20),
    st.text(min_size=1, max_size=5, alphabet=st.characters(min_codepoint=48, max_codepoint=57))
)
def test_trustedhost_ipv6_parsing(ipv6, port):
    host_with_port = f"[{ipv6}]:{port}"

    host_extracted = host_with_port.split(":")[0]

    if "[" in host_with_port:
        bracket_close = host_with_port.index("]")
        correct_host = host_with_port[1:bracket_close]

        assert host_extracted == f"[{ipv6}]" or host_extracted == ipv6, (
            f"IPv6 address should be correctly extracted"
        )
```

**Failing input**: Any IPv6 address with a port, e.g., `[::1]:8000`, `[2001:db8::1]:8080`

## Reproducing the Bug

```python
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.datastructures import Headers


async def dummy_app(scope, receive, send):
    pass


middleware = TrustedHostMiddleware(
    app=dummy_app,
    allowed_hosts=["::1"],
)

scope = {
    "type": "http",
    "headers": [[b"host", b"[::1]:8000"]],
}

headers = Headers(scope=scope)
host = headers.get("host", "").split(":")[0]

print(f"Host header: [::1]:8000")
print(f"Extracted: {host}")
print(f"Expected: [::1] or ::1")
```

Output:
```
Host header: [::1]:8000
Extracted: [
Expected: [::1] or ::1
```

The middleware extracts `[` instead of `::1`, causing valid IPv6 requests to be rejected as having an invalid host.

## Why This Is A Bug

IPv6 addresses are a fundamental part of modern networking. According to RFC 3986 and RFC 2732, IPv6 addresses in URIs/headers must be enclosed in brackets. The format is `[IPv6]:port`, e.g., `[2001:db8::1]:8080`.

The current code at line 40 in `trustedhost.py`:
```python
host = headers.get("host", "").split(":")[0]
```

This naively splits on the first `:`, which breaks for IPv6 addresses since they contain multiple colons.

Real-world impact:
- Users with IPv6-only networks cannot use TrustedHostMiddleware
- Dual-stack deployments may fail when clients connect via IPv6
- Security implications: IPv6 requests are always rejected, potentially causing DoS for IPv6 users

## Fix

```diff
--- a/starlette/middleware/trustedhost.py
+++ b/starlette/middleware/trustedhost.py
@@ -37,7 +37,14 @@ class TrustedHostMiddleware:
             return

         headers = Headers(scope=scope)
-        host = headers.get("host", "").split(":")[0]
+        host_header = headers.get("host", "")
+
+        # Handle IPv6 addresses in brackets [IPv6]:port
+        if host_header.startswith("["):
+            bracket_end = host_header.find("]")
+            host = host_header[1:bracket_end] if bracket_end != -1 else host_header
+        else:
+            host = host_header.split(":")[0]
+
         is_valid_host = False
         found_www_redirect = False
         for pattern in self.allowed_hosts:
```

This fix:
1. Checks if the host header starts with `[` (IPv6 format)
2. If yes, extracts the content between `[` and `]`
3. Otherwise, uses the original split logic for IPv4/hostnames