# Bug Report: django.utils.http.is_same_domain Case Insensitivity

**Target**: `django.utils.http.is_same_domain`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `is_same_domain` function incorrectly handles case sensitivity in domain comparisons. Since DNS is case-insensitive, the function should treat domains like "EXAMPLE.COM" and "example.com" as equivalent, but it currently returns False for such comparisons.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.utils.http import is_same_domain

@given(st.text(min_size=1))
def test_is_same_domain_exact_match(domain):
    result = is_same_domain(domain, domain)
    assert result is True
```

**Failing input**: `domain='A'` (or any uppercase domain)

## Reproducing the Bug

```python
from django.utils.http import is_same_domain

assert is_same_domain('EXAMPLE.COM', 'EXAMPLE.COM') == False
assert is_same_domain('Example.COM', 'example.com') == False
assert is_same_domain('FOO.EXAMPLE.COM', '.example.com') == False
```

## Why This Is A Bug

DNS is case-insensitive per RFC 4343. Domains should be compared case-insensitively. The function currently lowercases the `pattern` parameter but not the `host` parameter, leading to asymmetric comparisons where uppercase domains don't match themselves or their lowercase equivalents.

This is a security-relevant function used for domain validation (e.g., in CORS checks), so incorrect behavior could lead to security issues.

## Fix

```diff
--- a/django/utils/http.py
+++ b/django/utils/http.py
@@ -232,6 +232,7 @@ def is_same_domain(host, pattern):
     if not pattern:
         return False

+    host = host.lower()
     pattern = pattern.lower()
     return (
         pattern[0] == "."
```