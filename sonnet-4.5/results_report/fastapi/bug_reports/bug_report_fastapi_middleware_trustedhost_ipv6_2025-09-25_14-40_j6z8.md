# Bug Report: TrustedHostMiddleware Incorrectly Parses IPv6 Addresses

**Target**: `starlette.middleware.trustedhost.TrustedHostMiddleware.__call__` (re-exported as `fastapi.middleware.TrustedHostMiddleware`)
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The TrustedHostMiddleware incorrectly extracts the hostname from the Host header when the header contains an IPv6 address, causing all IPv6-based requests to be rejected even when they should be allowed.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.datastructures import Headers


@given(
    st.integers(min_value=1, max_value=65535)
)
def test_ipv6_host_parsing(port):
    class MockApp:
        async def __call__(self, scope, receive, send):
            pass

    middleware = TrustedHostMiddleware(
        MockApp(),
        allowed_hosts=["[::1]"]
    )

    host_with_port = f"[::1]:{port}"
    headers = Headers({"host": host_with_port})

    host = headers.get("host", "").split(":")[0]

    assert host == "[::1]", \
        f"Expected '[::1]' but got '{host}' when parsing '{host_with_port}'"
```

**Failing input**: Any port value (e.g., `port=8080`)

## Reproducing the Bug

```python
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.datastructures import Headers

test_cases = [
    "[::1]",
    "[::1]:8080",
    "[2001:db8::1]",
    "[2001:db8::1]:8080",
]

for host_header in test_cases:
    host = host_header.split(":")[0]
    print(f"Input: '{host_header}'")
    print(f"Extracted host: '{host}'")
    print(f"Expected: IPv6 address in brackets")
    print()
```

**Output**:
```
Input: '[::1]'
Extracted host: '['
Expected: IPv6 address in brackets

Input: '[::1]:8080'
Extracted host: '['
Expected: IPv6 address in brackets

Input: '[2001:db8::1]'
Extracted host: '[2001'
Expected: IPv6 address in brackets

Input: '[2001:db8::1]:8080'
Extracted host: '[2001'
Expected: IPv6 address in brackets
```

## Why This Is A Bug

HTTP Host headers containing IPv6 addresses follow the format `[ipv6]:port` where the IPv6 address is enclosed in square brackets (per RFC 3986). The current implementation at line 40 of `starlette/middleware/trustedhost.py`:

```python
host = headers.get("host", "").split(":")[0]
```

This code incorrectly assumes that splitting by `:` will separate the hostname from the port. However, IPv6 addresses contain multiple colons, so this approach fails:

- `"[::1]:8080".split(":")[0]` returns `"["` instead of `"[::1]"`
- `"[2001:db8::1]".split(":")[0]` returns `"[2001"` instead of `"[2001:db8::1]"`

As a result:
1. The extracted "host" value is malformed and won't match any allowed host patterns
2. All requests using IPv6 addresses will be rejected with 400 status, even if the IPv6 address is in the allowed hosts list
3. The www redirect logic (line 47) also fails for IPv6 addresses

## Fix

```diff
--- a/starlette/middleware/trustedhost.py
+++ b/starlette/middleware/trustedhost.py
@@ -37,7 +37,15 @@ class TrustedHostMiddleware:
             return

         headers = Headers(scope=scope)
-        host = headers.get("host", "").split(":")[0]
+        host_header = headers.get("host", "")
+
+        # Handle IPv6 addresses in brackets, e.g., [::1]:port or [::1]
+        if host_header.startswith("["):
+            # IPv6 address: extract everything up to and including the closing bracket
+            bracket_end = host_header.find("]")
+            host = host_header[:bracket_end + 1] if bracket_end != -1 else host_header
+        else:
+            # IPv4 or hostname: split by colon to remove port
+            host = host_header.split(":")[0]
+
         is_valid_host = False
         found_www_redirect = False
```

This fix:
1. Checks if the host header starts with `[` to identify IPv6 addresses
2. For IPv6 addresses, extracts everything up to and including the closing `]`
3. For IPv4 addresses and hostnames, uses the original split logic
4. Correctly handles all valid Host header formats per RFC 3986