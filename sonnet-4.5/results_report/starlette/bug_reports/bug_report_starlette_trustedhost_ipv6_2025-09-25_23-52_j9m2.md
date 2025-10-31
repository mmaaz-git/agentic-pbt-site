# Bug Report: TrustedHostMiddleware IPv6 Host Parsing

**Target**: `starlette.middleware.trustedhost.TrustedHostMiddleware`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

TrustedHostMiddleware incorrectly parses IPv6 addresses when a port is specified in the Host header. The middleware uses `split(":")` to separate the host from the port, which breaks IPv6 addresses (e.g., `[::1]:8000`) because IPv6 addresses contain multiple colons.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.datastructures import Headers


@given(st.integers(min_value=1, max_value=65535))
def test_trustedhost_ipv6_port_handling(port):
    middleware = TrustedHostMiddleware(
        app=lambda s, r, sn: None,
        allowed_hosts=["[::1]"]
    )

    host_header = f"[::1]:{port}"
    extracted_host = host_header.split(":")[0]

    assert extracted_host == "[::1]", (
        f"IPv6 host should be '[::1]' but got '{extracted_host}'"
    )
```

**Failing input**: Any port number (e.g., `port=8000`)

## Reproducing the Bug

```python
from starlette.middleware.trustedhost import TrustedHostMiddleware

middleware = TrustedHostMiddleware(
    app=lambda s, r, sn: None,
    allowed_hosts=["[::1]"]
)

host_header = "[::1]:8000"
extracted_host = host_header.split(":")[0]

print(f"Host header: {host_header}")
print(f"Extracted host: {extracted_host}")
print(f"Expected: [::1]")
print(f"Actual: {extracted_host}")
```

**Output:**
```
Host header: [::1]:8000
Extracted host: [
Expected: [::1]
Actual: [
```

## Why This Is A Bug

Line 40 in `trustedhost.py` uses:
```python
host = headers.get("host", "").split(":")[0]
```

This approach works for regular hostnames with ports (e.g., `example.com:8000` → `example.com`), but fails for IPv6 addresses because:
- IPv6 addresses contain multiple colons: `[::1]`, `[2001:db8::1]`
- When enclosed in brackets with a port: `[::1]:8000`
- Splitting by `:` and taking `[0]` gives `[` instead of `[::1]`

As a result, any IPv6 address with a port will be incorrectly extracted and will always be rejected by the middleware, even if it's in the `allowed_hosts` list.

## Fix

```diff
--- a/starlette/middleware/trustedhost.py
+++ b/starlette/middleware/trustedhost.py
@@ -37,7 +37,11 @@ class TrustedHostMiddleware:
             return

         headers = Headers(scope=scope)
-        host = headers.get("host", "").split(":")[0]
+        host_header = headers.get("host", "")
+        # IPv6 addresses are enclosed in brackets like [::1]:8000
+        if host_header.startswith("["):
+            host = host_header.rsplit(":", 1)[0]
+        else:
+            host = host_header.split(":", 1)[0]
         is_valid_host = False
         found_www_redirect = False
         for pattern in self.allowed_hosts:
```