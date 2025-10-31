# Bug Report: TrustedHostMiddleware IPv6 Address Parsing

**Target**: `starlette.middleware.trustedhost.TrustedHostMiddleware`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

TrustedHostMiddleware incorrectly parses IPv6 addresses from the Host header, causing host validation to fail for legitimate IPv6 requests.

## Property-Based Test

```python
from hypothesis import given, strategies as st, example
from starlette.datastructures import Headers

@given(st.sampled_from(["::1", "2001:db8::1", "fe80::1"]), st.integers(min_value=1, max_value=65535))
@example("::1", 8080)
def test_ipv6_host_parsing(ipv6_addr, port):
    host_header = f"[{ipv6_addr}]:{port}"

    scope = {
        'type': 'http',
        'headers': [(b'host', host_header.encode())]
    }

    headers = Headers(scope=scope)
    parsed_host = headers.get("host", "").split(":")[0]

    assert parsed_host == ipv6_addr, f"Expected {ipv6_addr}, got {parsed_host}"
```

**Failing input**: `["::1", 8080]`

## Reproducing the Bug

```python
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.datastructures import Headers

def dummy_app(scope, receive, send):
    pass

middleware = TrustedHostMiddleware(dummy_app, allowed_hosts=["::1"])

scope = {
    'type': 'http',
    'headers': [(b'host', b'[::1]:8080')]
}

headers = Headers(scope=scope)
host = headers.get("host", "").split(":")[0]

print(f"Host header: [::1]:8080")
print(f"Parsed host: {host}")
print(f"Expected: ::1")
print(f"Actual: [")
```

Output:
```
Host header: [::1]:8080
Parsed host: [
Expected: ::1
Actual: [
```

## Why This Is A Bug

IPv6 addresses are formatted in the Host header as `[address]:port` per RFC 3986. The current implementation splits on `:` to extract the hostname, which fails for IPv6 because IPv6 addresses contain multiple colons.

For example:
- `[::1]:8080` → split(":")[0] = `[` (wrong, should be `::1`)
- `[2001:db8::1]:80` → split(":")[0] = `[2001` (wrong, should be `2001:db8::1`)

This causes TrustedHostMiddleware to incorrectly reject valid IPv6 requests or fail to match them against allowed hosts.

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
+            host = host_header.split("]")[0][1:]
+        else:
+            host = host_header.split(":")[0]
+
         is_valid_host = False
         found_www_redirect = False
         for pattern in self.allowed_hosts:
```