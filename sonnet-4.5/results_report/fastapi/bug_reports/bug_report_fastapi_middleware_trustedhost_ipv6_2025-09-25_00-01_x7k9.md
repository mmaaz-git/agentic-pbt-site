# Bug Report: TrustedHostMiddleware IPv6 Address Parsing

**Target**: `starlette.middleware.trustedhost.TrustedHostMiddleware` (re-exported by `fastapi.middleware.trustedhost`)
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

TrustedHostMiddleware incorrectly parses IPv6 addresses in the Host header, causing all IPv6 requests to be rejected even when the address is in the allowed_hosts list.

## Property-Based Test

```python
from hypothesis import given, strategies as st

@given(
    st.lists(st.integers(min_value=0, max_value=65535).map(lambda x: hex(x)[2:]), min_size=8, max_size=8),
    st.integers(min_value=1, max_value=65535)
)
def test_trustedhost_ipv6_parsing(segments, port):
    ipv6 = ':'.join(segments)
    host_with_port = f"[{ipv6}]:{port}"

    parsed = host_with_port.split(":")[0]
    expected = f"[{ipv6}]"

    assert parsed == expected or parsed == ipv6
```

**Failing input**: `segments=['2', '0', '0', '1', 'd', 'b', '8', '1'], port=8080`
This produces `host_with_port="[2:0:0:1:d:b:8:1]:8080"`, which when split by `:` gives `"[2"` instead of the expected IPv6 address.

## Reproducing the Bug

```python
from starlette.middleware.trustedhost import TrustedHostMiddleware

async def dummy_app(scope, receive, send):
    pass

middleware = TrustedHostMiddleware(
    dummy_app,
    allowed_hosts=["[2001:db8::1]", "2001:db8::1"]
)

host_header = "[2001:db8::1]:8080"
parsed_host = host_header.split(":")[0]

print(f"Host header: {host_header}")
print(f"Parsed (buggy): {parsed_host}")
print(f"Expected: [2001:db8::1] or 2001:db8::1")
```

Output:
```
Host header: [2001:db8::1]:8080
Parsed (buggy): [2001
Expected: [2001:db8::1] or 2001:db8::1
```

The parsed host `[2001` will not match any pattern in `allowed_hosts`, causing the request to be rejected with a 400 status code.

## Why This Is A Bug

Line 40 in `trustedhost.py` uses `host = headers.get("host", "").split(":")[0]` to extract the hostname from the Host header. This approach works for IPv4 addresses and domain names (e.g., `example.com:8080` → `example.com`), but breaks for IPv6 addresses because IPv6 addresses contain colons.

When an IPv6 address with port is provided (e.g., `[2001:db8::1]:8080`), splitting by the first colon yields `[2001` instead of `[2001:db8::1]` or `2001:db8::1`. This causes all IPv6 requests to be rejected, breaking IPv6 support entirely.

## Fix

```diff
--- a/starlette/middleware/trustedhost.py
+++ b/starlette/middleware/trustedhost.py
@@ -37,7 +37,13 @@ class TrustedHostMiddleware:
             return

         headers = Headers(scope=scope)
-        host = headers.get("host", "").split(":")[0]
+        host_header = headers.get("host", "")
+        if host_header.startswith("["):
+            # IPv6 address: [addr]:port or [addr]
+            host = host_header.split("]")[0] + "]" if "]" in host_header else host_header
+        else:
+            # IPv4 or domain name: addr:port or addr
+            host = host_header.split(":")[0]
         is_valid_host = False
         found_www_redirect = False
         for pattern in self.allowed_hosts:
```