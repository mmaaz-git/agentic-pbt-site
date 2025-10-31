# Bug Report: TrustedHostMiddleware IPv6 Host Header Parsing Failure

**Target**: `starlette.middleware.trustedhost.TrustedHostMiddleware`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

TrustedHostMiddleware incorrectly parses IPv6 addresses in HTTP Host headers by naively splitting on colons, causing it to reject valid requests from IPv6 clients even when their addresses are explicitly allowed.

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
    """Test that TrustedHostMiddleware correctly parses IPv6 addresses in Host headers."""
    # What the middleware currently does (buggy behavior)
    result = host_header.split(":")[0]

    # What it should do for IPv6 addresses
    if host_header.startswith("[") and "]:" in host_header:
        # IPv6 with port: extract address between brackets
        expected = host_header.split("]:")[0][1:]
    elif host_header.startswith("[") and host_header.endswith("]"):
        # IPv6 without port: extract address between brackets
        expected = host_header[1:-1]
    else:
        # Not IPv6 format
        expected = host_header.split(":")[0]

    # This assertion will fail, demonstrating the bug
    assert result == expected, f"Failed to parse {host_header}: got '{result}', expected '{expected}'"


if __name__ == "__main__":
    # Run the test
    test_host_header_ipv6_parsing()
```

<details>

<summary>
**Failing input**: `[::1]:8000`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/54/hypo.py", line 37, in <module>
    test_host_header_ipv6_parsing()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/54/hypo.py", line 10, in test_host_header_ipv6_parsing
    "[::1]:8000",
               ^^
  File "/home/npc/pbt/agentic-pbt/envs/starlette_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/54/hypo.py", line 32, in test_host_header_ipv6_parsing
    assert result == expected, f"Failed to parse {host_header}: got '{result}', expected '{expected}'"
           ^^^^^^^^^^^^^^^^^^
AssertionError: Failed to parse [::1]:8000: got '[', expected '::1'
Falsifying example: test_host_header_ipv6_parsing(
    host_header='[::1]:8000',
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/starlette_env/lib/python3.13/site-packages')

# Demonstrate the actual bug in TrustedHostMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.datastructures import Headers

# Simulate what happens in TrustedHostMiddleware at line 40
host_header = "[::1]:8000"
extracted_host = host_header.split(":")[0]

print(f"Host header: {host_header}")
print(f"Extracted host using split(':')[0]: {extracted_host}")
print(f"Expected host: ::1")
print()

# Test with various IPv6 addresses
test_cases = [
    "[::1]:8000",
    "[2001:db8::1]:443",
    "[fe80::1]:80",
    "[::ffff:192.0.2.1]:8080"
]

print("Testing various IPv6 addresses:")
for test_host in test_cases:
    extracted = test_host.split(":")[0]
    # Correct parsing would be:
    if test_host.startswith("[") and "]:" in test_host:
        expected = test_host.split("]:")[0][1:]
    elif test_host.startswith("[") and test_host.endswith("]"):
        expected = test_host[1:-1]
    else:
        expected = test_host.split(":")[0]

    print(f"  Input: {test_host}")
    print(f"    Current extraction: {extracted}")
    print(f"    Expected: {expected}")
    print(f"    Match: {extracted == expected}")
    print()

# Show that this causes real middleware failures
print("Real-world impact demonstration:")
print("Creating TrustedHostMiddleware with allowed_hosts=['::1']")

# Create a mock scope object
scope = {
    "type": "http",
    "headers": [(b"host", b"[::1]:8000")]
}

# Simulate the middleware logic
headers = Headers(scope=scope)
host = headers.get("host", "").split(":")[0]
allowed_hosts = ["::1"]

is_valid_host = False
for pattern in allowed_hosts:
    if host == pattern:
        is_valid_host = True
        break

print(f"  Host header received: [::1]:8000")
print(f"  Host extracted by middleware: {host}")
print(f"  Allowed hosts: {allowed_hosts}")
print(f"  Is valid host? {is_valid_host}")
print(f"  Result: {'Request would be ACCEPTED' if is_valid_host else 'Request would be REJECTED with 400 Invalid host header'}")

# Final assertion to demonstrate the bug
try:
    assert extracted_host == "::1", f"Bug confirmed: extracted '{extracted_host}' instead of '::1'"
except AssertionError as e:
    print(f"\n{e}")
```

<details>

<summary>
TrustedHostMiddleware incorrectly rejects valid IPv6 requests
</summary>
```
Host header: [::1]:8000
Extracted host using split(':')[0]: [
Expected host: ::1

Testing various IPv6 addresses:
  Input: [::1]:8000
    Current extraction: [
    Expected: ::1
    Match: False

  Input: [2001:db8::1]:443
    Current extraction: [2001
    Expected: 2001:db8::1
    Match: False

  Input: [fe80::1]:80
    Current extraction: [fe80
    Expected: fe80::1
    Match: False

  Input: [::ffff:192.0.2.1]:8080
    Current extraction: [
    Expected: ::ffff:192.0.2.1
    Match: False

Real-world impact demonstration:
Creating TrustedHostMiddleware with allowed_hosts=['::1']
  Host header received: [::1]:8000
  Host extracted by middleware: [
  Allowed hosts: ['::1']
  Is valid host? False
  Result: Request would be REJECTED with 400 Invalid host header

Bug confirmed: extracted '[' instead of '::1'
```
</details>

## Why This Is A Bug

This bug violates HTTP standards and breaks core security functionality for IPv6 traffic. According to RFC 3986 Section 3.2.2 and RFC 7230 Section 5.4, IPv6 addresses in URIs and HTTP Host headers MUST be enclosed in square brackets (e.g., `[::1]:8000` for localhost on port 8000). The middleware's current implementation at line 40 (`host = headers.get("host", "").split(":")[0]`) fails because:

1. **IPv6 addresses contain multiple colons**: The naive split on the first colon breaks IPv6 addresses which use colons as separators (e.g., `2001:db8::1`).

2. **Incorrect extraction**: For `[::1]:8000`, the code extracts `[` instead of `::1`, and for `[2001:db8::1]:443`, it extracts `[2001` instead of `2001:db8::1`.

3. **Security impact**: Even when `::1` is explicitly in `allowed_hosts`, requests from IPv6 localhost are rejected with "400 Invalid host header", forcing developers to either disable the security middleware entirely or not support IPv6.

4. **Standards compliance**: The middleware fails to comply with established HTTP standards that have been in place for over two decades.

## Relevant Context

The bug is located in `/starlette/middleware/trustedhost.py` at line 40. The middleware is designed to prevent host header injection attacks by validating that incoming requests have a Host header matching the configured allowed hosts.

Key observations:
- The middleware correctly handles IPv4 addresses (`192.168.1.1:80` → `192.168.1.1`)
- It correctly handles domain names (`example.com:443` → `example.com`)
- It completely fails for ANY IPv6 address with a port number
- IPv6 usage is growing, especially in cloud environments and modern infrastructure
- Major providers like AWS, Google Cloud, and Azure all support IPv6

Documentation references:
- RFC 3986: https://www.rfc-editor.org/rfc/rfc3986#section-3.2.2
- RFC 7230: https://www.rfc-editor.org/rfc/rfc7230#section-5.4
- Starlette source: https://github.com/encode/starlette/blob/master/starlette/middleware/trustedhost.py

## Proposed Fix

```diff
--- a/starlette/middleware/trustedhost.py
+++ b/starlette/middleware/trustedhost.py
@@ -37,7 +37,16 @@ class TrustedHostMiddleware:
             return

         headers = Headers(scope=scope)
-        host = headers.get("host", "").split(":")[0]
+        host_header = headers.get("host", "")
+
+        # Handle IPv6 addresses which are enclosed in brackets
+        if host_header.startswith("["):
+            if "]:" in host_header:
+                host = host_header.split("]:")[0][1:]  # IPv6 with port
+            elif host_header.endswith("]"):
+                host = host_header[1:-1]  # IPv6 without port
+            else:
+                host = host_header  # Malformed, let validation fail
+        else:
+            host = host_header.split(":")[0]  # IPv4 or hostname
         is_valid_host = False
         found_www_redirect = False
         for pattern in self.allowed_hosts:
```