# Bug Report: django.utils.http.is_same_domain Case Sensitivity

**Target**: `django.utils.http.is_same_domain`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `is_same_domain()` function fails to perform case-insensitive domain matching when the host parameter contains uppercase letters. The function only lowercases the pattern but not the host, causing legitimate domain matches to fail.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings as hyp_settings
from django.utils.http import is_same_domain

@given(st.text(min_size=1))
@hyp_settings(max_examples=500)
def test_is_same_domain_case_insensitive(host):
    """Property: Domain matching should be case-insensitive"""
    pattern = host.upper()
    result1 = is_same_domain(host.lower(), pattern)
    result2 = is_same_domain(host.upper(), pattern.lower())
    assert result1 == result2, \
        f"Case sensitivity mismatch: is_same_domain({host.lower()!r}, {pattern!r}) = {result1}, " \
        f"but is_same_domain({host.upper()!r}, {pattern.lower()!r}) = {result2}"
```

**Failing input**: `host='A'`

## Reproducing the Bug

```python
from django.utils.http import is_same_domain

assert is_same_domain('A', 'A') == False
assert is_same_domain('example.COM', 'EXAMPLE.com') == False
assert is_same_domain('Example.Com', 'example.com') == False

print("is_same_domain('A', 'A') =", is_same_domain('A', 'A'))
print("is_same_domain('example.COM', 'EXAMPLE.com') =", is_same_domain('example.COM', 'EXAMPLE.com'))
```

Output:
```
is_same_domain('A', 'A') = False
is_same_domain('example.COM', 'EXAMPLE.com') = False
```

## Why This Is A Bug

The function's docstring states "Any pattern beginning with a period matches a domain and all of its subdomains", implying domain matching should follow standard DNS rules where domain names are case-insensitive (RFC 1035). The function already attempts case-insensitive matching by calling `pattern.lower()` on line 235, but fails to also lowercase the `host` parameter.

This bug causes:
1. Security vulnerabilities: CORS/CSRF protection may fail if the Host header uses different casing
2. Incorrect domain validation in security-sensitive contexts
3. Asymmetric behavior: `is_same_domain('a', 'A')` returns True but `is_same_domain('A', 'a')` returns False

## Fix

```diff
--- a/django/utils/http.py
+++ b/django/utils/http.py
@@ -232,7 +232,8 @@ def is_same_domain(host, pattern):
     if not pattern:
         return False

     pattern = pattern.lower()
+    host = host.lower()
     return (
         pattern[0] == "."
         and (host.endswith(pattern) or host == pattern[1:])
         or pattern == host
     )
```