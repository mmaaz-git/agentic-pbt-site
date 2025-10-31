# Bug Report: TrustedHostMiddleware IPv6 Host Parsing

**Target**: `starlette.middleware.trustedhost.TrustedHostMiddleware`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

TrustedHostMiddleware incorrectly parses IPv6 addresses in the HTTP Host header by splitting on `:`, which breaks IPv6 addresses that contain colons as part of their format.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/starlette_env/lib/python3.13/site-packages')

import pytest
from hypothesis import given, settings, example
import hypothesis.strategies as st


@given(st.sampled_from([
    "[::1]:8000",
    "[2001:db8::1]:443",
    "[fe80::1]:80",
    "[::ffff:192.0.2.1]:8080"
]))
def test_host_header_ipv6_parsing(host_header):
    result = host_header.split(":")[0]

    if ']:' in host_header:
        expected = host_header.split(']:')[0][1:]
        assert result == expected, f"Failed to parse {host_header}: got {result}, expected {expected}"
```

**Failing input**: `[::1]:8000` (and all other IPv6 addresses with ports)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/starlette_env/lib/python3.13/site-packages')

host_header = "[::1]:8000"
extracted_host = host_header.split(":")[0]

print(f"Host header: {host_header}")
print(f"Extracted host: {extracted_host}")
print(f"Expected: ::1")
print(f"Got: {extracted_host}")

assert extracted_host == "::1"
```

Output:
```
Host header: [::1]:8000
Extracted host: [
Expected: ::1
Got: [
AssertionError
```

## Why This Is A Bug

According to RFC 3986, IPv6 addresses in URLs must be enclosed in brackets. For example, `http://[::1]:8000/` represents localhost on port 8000. The HTTP Host header follows the same convention: `[::1]:8000`.

The current implementation at line 40 of `trustedhost.py`:
```python
host = headers.get("host", "").split(":")[0]
```

This works for IPv4 addresses and hostnames (e.g., `example.com:8000` → `example.com`), but fails for IPv6 addresses because they contain multiple colons.

Examples of incorrect parsing:
- `[::1]:8000` → `[` (should be `::1`)
- `[2001:db8::1]:443` → `[2001` (should be `2001:db8::1`)
- `[fe80::1]:80` → `[fe80` (should be `fe80::1`)

This causes the middleware to:
1. Fail to match valid IPv6 hosts against allowed_hosts patterns
2. Incorrectly validate or reject requests from IPv6 clients

## Fix

```diff
--- a/starlette/middleware/trustedhost.py
+++ b/starlette/middleware/trustedhost.py
@@ -37,7 +37,16 @@ class TrustedHostMiddleware:
             return

         headers = Headers(scope=scope)
-        host = headers.get("host", "").split(":")[0]
+        host_header = headers.get("host", "")
+
+        if host_header.startswith("["):
+            if "]:" in host_header:
+                host = host_header.split("]:")[0][1:]
+            elif host_header.endswith("]"):
+                host = host_header[1:-1]
+            else:
+                host = host_header
+        else:
+            host = host_header.split(":")[0]
         is_valid_host = False
         found_www_redirect = False
         for pattern in self.allowed_hosts:
```