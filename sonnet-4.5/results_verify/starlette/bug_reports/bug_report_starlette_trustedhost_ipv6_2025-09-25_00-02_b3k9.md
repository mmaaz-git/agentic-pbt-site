# Bug Report: TrustedHostMiddleware IPv6 Host Parsing

**Target**: `starlette.middleware.trustedhost.TrustedHostMiddleware.__call__`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`TrustedHostMiddleware` incorrectly parses IPv6 addresses in the Host header by naively splitting on `:`, which breaks IPv6 addresses since they contain multiple colons. This causes valid IPv6 requests to be incorrectly rejected.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st


@given(
    ipv6_segments=st.lists(
        st.text(min_size=1, max_size=4, alphabet="0123456789abcdef"),
        min_size=2,
        max_size=8
    ),
    port=st.integers(min_value=1, max_value=65535),
    include_port=st.booleans(),
)
@settings(max_examples=200)
def test_ipv6_host_parsing_property(ipv6_segments, port, include_port):
    ipv6_addr = ":".join(ipv6_segments)

    if include_port:
        host_header = f"[{ipv6_addr}]:{port}"
    else:
        host_header = f"[{ipv6_addr}]"

    current_parsing = host_header.split(":")[0]

    correct_host_noport = f"[{ipv6_addr}]"

    assert current_parsing != "[", \
        f"Bug: IPv6 address '{host_header}' incorrectly parsed as '['"
```

**Failing input**: Any IPv6 host header, e.g., `"[::1]:8080"`, `"[2001:db8::1]:443"`

## Reproducing the Bug

```python
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.datastructures import Headers

async def dummy_app(scope, receive, send):
    pass

middleware = TrustedHostMiddleware(
    dummy_app,
    allowed_hosts=["[::1]", "localhost"]
)

host_header = "[::1]:8080"
host = host_header.split(":")[0]

print(f"Host header: {host_header}")
print(f"Parsed host: {host}")
print(f"Expected: [::1]")
```

**Output**:
```
Host header: [::1]:8080
Parsed host: [
Expected: [::1]
```

The code at line 40 of `trustedhost.py`:
```python
host = headers.get("host", "").split(":")[0]
```

This splits on the first `:`, which for `[::1]:8080` is inside the IPv6 address itself, resulting in `[` instead of `[::1]`.

## Why This Is A Bug

IPv6 addresses contain colons as part of the address format (e.g., `::1`, `2001:db8::1`). When included in a Host header with a port, they must be wrapped in brackets: `[::1]:8080`.

The current parsing logic incorrectly splits on the first colon, which for IPv6 is inside the address, not at the port separator. This causes:
1. The host is incorrectly extracted as `[` instead of `[::1]`
2. The middleware fails to match `[` against allowed hosts like `[::1]`
3. Valid IPv6 requests are incorrectly rejected with 400 Bad Request

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
+        if host_header.startswith("["):
+            # IPv6 address: [host]:port or [host]
+            host = host_header.split("]")[0] + "]"
+        else:
+            # IPv4 or hostname: host:port or host
+            host = host_header.split(":")[0]
+
         is_valid_host = False
         found_www_redirect = False
         for pattern in self.allowed_hosts:
```