# Bug Report: TrustedHostMiddleware Wildcard Pattern Matching

**Target**: `starlette.middleware.trustedhost.TrustedHostMiddleware`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The TrustedHostMiddleware wildcard pattern matching incorrectly accepts hostnames that start with a dot (e.g., `.example.com`) when using wildcard patterns like `*.example.com`. This allows invalid hostnames to bypass host validation.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from starlette.middleware.trustedhost import TrustedHostMiddleware


def dummy_app(scope, receive, send):
    pass


@given(st.text(alphabet="abcdefghijklmnopqrstuvwxyz.", min_size=1, max_size=20))
def test_trustedhost_no_dot_prefix(domain):
    middleware = TrustedHostMiddleware(dummy_app, allowed_hosts=[f"*.{domain}"])

    pattern = f"*.{domain}"
    invalid_host = f".{domain}"

    is_valid_host = False
    for p in middleware.allowed_hosts:
        if invalid_host == p or (p.startswith("*") and invalid_host.endswith(p[1:])):
            is_valid_host = True
            break

    assert is_valid_host == False, f"Invalid host '{invalid_host}' should not match pattern '{pattern}'"
```

**Failing input**: `domain="example.com"` (or any valid domain)

## Reproducing the Bug

```python
from starlette.middleware.trustedhost import TrustedHostMiddleware


def dummy_app(scope, receive, send):
    pass


middleware = TrustedHostMiddleware(dummy_app, allowed_hosts=["*.example.com"])

pattern = "*.example.com"
invalid_host = ".example.com"

is_valid_host = False
for p in middleware.allowed_hosts:
    if invalid_host == p or (p.startswith("*") and invalid_host.endswith(p[1:])):
        is_valid_host = True
        break

assert is_valid_host == True
```

## Why This Is A Bug

The pattern matching logic on line 44 of `trustedhost.py` uses:

```python
if host == pattern or (pattern.startswith("*") and host.endswith(pattern[1:])):
```

When the pattern is `*.example.com`, `pattern[1:]` is `.example.com`. The condition `host.endswith(pattern[1:])` evaluates to `True` for the invalid host `.example.com` because it literally ends with `.example.com` (it *is* `.example.com`).

However, a hostname starting with a dot is invalid according to DNS standards and should not match the wildcard pattern. The wildcard `*.example.com` should only match valid subdomains like `sub.example.com`, not malformed hostnames like `.example.com`.

This could be a security issue if an attacker can craft HTTP requests with such malformed Host headers to bypass host validation.

## Fix

```diff
--- a/starlette/middleware/trustedhost.py
+++ b/starlette/middleware/trustedhost.py
@@ -41,7 +41,9 @@ class TrustedHostMiddleware:
         is_valid_host = False
         found_www_redirect = False
         for pattern in self.allowed_hosts:
-            if host == pattern or (pattern.startswith("*") and host.endswith(pattern[1:])):
+            if host == pattern or (
+                pattern.startswith("*") and host.endswith(pattern[1:]) and not host.startswith(".")
+            ):
                 is_valid_host = True
                 break
             elif "www." + host == pattern:
```