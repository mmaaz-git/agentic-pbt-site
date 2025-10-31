# Bug Report: django.utils.http.is_same_domain Case Sensitivity Violation

**Target**: `django.utils.http.is_same_domain`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `is_same_domain()` function violates DNS RFC 1035 by performing case-sensitive domain matching when the host contains uppercase letters. The function only lowercases the pattern but fails to lowercase the host parameter.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Hypothesis property-based test for django.utils.http.is_same_domain case sensitivity"""

# Add Django environment to path
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

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

if __name__ == "__main__":
    test_is_same_domain_case_insensitive()
```

<details>

<summary>
**Failing input**: `host='A'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/29/hypo.py", line 23, in <module>
    test_is_same_domain_case_insensitive()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/29/hypo.py", line 12, in test_is_same_domain_case_insensitive
    @hyp_settings(max_examples=500)
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/29/hypo.py", line 18, in test_is_same_domain_case_insensitive
    assert result1 == result2, \
           ^^^^^^^^^^^^^^^^^^
AssertionError: Case sensitivity mismatch: is_same_domain('a', 'A') = True, but is_same_domain('A', 'a') = False
Falsifying example: test_is_same_domain_case_insensitive(
    host='A',
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction of django.utils.http.is_same_domain case sensitivity bug"""

# Add Django environment to path
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.utils.http import is_same_domain

# Test basic uppercase domains - should return True but returns False
print("Test 1: is_same_domain('A', 'A') =", is_same_domain('A', 'A'))
print("Expected: True, Got:", is_same_domain('A', 'A'))
print()

# Test mixed case domain matching - should return True but returns False
print("Test 2: is_same_domain('example.COM', 'EXAMPLE.com') =", is_same_domain('example.COM', 'EXAMPLE.com'))
print("Expected: True, Got:", is_same_domain('example.COM', 'EXAMPLE.com'))
print()

print("Test 3: is_same_domain('Example.Com', 'example.com') =", is_same_domain('Example.Com', 'example.com'))
print("Expected: True, Got:", is_same_domain('Example.Com', 'example.com'))
print()

# Show asymmetric behavior
print("Asymmetric behavior demonstration:")
print("  is_same_domain('a', 'A') =", is_same_domain('a', 'A'))  # Returns True
print("  is_same_domain('A', 'a') =", is_same_domain('A', 'a'))  # Returns False
print("This asymmetry shows only pattern is lowercased, not host")
print()

# Test subdomain matching with case sensitivity
print("Subdomain matching:")
print("  is_same_domain('sub.EXAMPLE.com', '.example.com') =", is_same_domain('sub.EXAMPLE.com', '.example.com'))
print("  is_same_domain('SUB.example.com', '.EXAMPLE.COM') =", is_same_domain('SUB.example.com', '.EXAMPLE.COM'))
print()

# Real-world scenario - HTTP Host headers can have any case
print("Real-world scenario (HTTP Host header with various cases):")
host_variations = ['Example.Com', 'EXAMPLE.COM', 'example.com', 'ExAmPlE.cOm']
pattern = 'example.com'
print(f"Pattern: {pattern}")
for host in host_variations:
    result = is_same_domain(host, pattern)
    print(f"  is_same_domain('{host}', '{pattern}') = {result}")
```

<details>

<summary>
Asymmetric behavior and real-world failures
</summary>
```
Test 1: is_same_domain('A', 'A') = False
Expected: True, Got: False

Test 2: is_same_domain('example.COM', 'EXAMPLE.com') = False
Expected: True, Got: False

Test 3: is_same_domain('Example.Com', 'example.com') = False
Expected: True, Got: False

Asymmetric behavior demonstration:
  is_same_domain('a', 'A') = True
  is_same_domain('A', 'a') = False
This asymmetry shows only pattern is lowercased, not host

Subdomain matching:
  is_same_domain('sub.EXAMPLE.com', '.example.com') = False
  is_same_domain('SUB.example.com', '.EXAMPLE.COM') = True

Real-world scenario (HTTP Host header with various cases):
Pattern: example.com
  is_same_domain('Example.Com', 'example.com') = False
  is_same_domain('EXAMPLE.COM', 'example.com') = False
  is_same_domain('example.com', 'example.com') = True
  is_same_domain('ExAmPlE.cOm', 'example.com') = False
```
</details>

## Why This Is A Bug

This violates expected behavior and standards in multiple ways:

1. **DNS RFC 1035 Violation**: Section 3.1 explicitly states "For all parts of the DNS that are part of the official protocol, comparisons between character strings (e.g., labels, domain names, etc.) are done in a case-insensitive manner." This is not optional - it's a requirement for DNS compliance.

2. **Security Vulnerability**: This function is used in Django's CSRF protection (`django.middleware.csrf`), CORS handling, and `ALLOWED_HOSTS` validation. When a browser sends a Host header like "Example.Com" and Django's ALLOWED_HOSTS contains "example.com", the validation fails incorrectly, potentially breaking legitimate requests or creating security gaps.

3. **Partial Implementation Bug**: The function already calls `pattern.lower()` on line 235, clearly showing the developers intended case-insensitive comparison. The omission of `host.lower()` is an obvious oversight, not intentional design.

4. **Asymmetric Behavior**: The function exhibits nonsensical asymmetric behavior where `is_same_domain('a', 'A')` returns `True` but `is_same_domain('A', 'a')` returns `False`. This violates the fundamental property that domain equality should be symmetric.

5. **Production Impact**: HTTP Host headers can arrive from browsers and proxies in any case combination. Major browsers like Chrome may preserve or modify case. This bug causes legitimate requests to fail when the Host header case doesn't exactly match the configured allowed hosts.

## Relevant Context

The `is_same_domain` function is located in `/django/utils/http.py` at lines 223-240. It's a critical security function used throughout Django:

- **CSRF Middleware**: Uses this to validate referer headers against allowed hosts
- **CORS Headers**: Validates origin domains for cross-origin requests
- **ALLOWED_HOSTS**: Core security feature that validates incoming Host headers
- **url_has_allowed_host_and_scheme**: Depends on this for redirect validation

The function's docstring doesn't explicitly mention case sensitivity, but given:
1. DNS standards mandate case-insensitive comparison
2. The code already lowercases the pattern (showing clear intent)
3. It's used for security-critical host validation

The missing `host.lower()` is clearly a bug, not a feature.

Relevant documentation:
- [RFC 1035 Section 3.1](https://www.rfc-editor.org/rfc/rfc1035#section-3.1) - DNS case insensitivity requirement
- [Django Security Documentation](https://docs.djangoproject.com/en/stable/topics/security/) - Describes ALLOWED_HOSTS and CSRF protection
- Source code: `django/utils/http.py:223-240`

## Proposed Fix

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