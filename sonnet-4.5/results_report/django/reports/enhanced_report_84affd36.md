# Bug Report: django.utils.http.is_same_domain Case Sensitivity Bug

**Target**: `django.utils.http.is_same_domain`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `is_same_domain` function incorrectly performs case-sensitive domain comparison, violating DNS RFC 1035 which mandates case-insensitive comparison for domain names. The function lowercases the `pattern` parameter but not the `host` parameter, causing identical domains with different casing to be incorrectly considered non-matching.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from django.utils.http import is_same_domain

@given(host=st.text(
    alphabet=st.characters(whitelist_categories=('Ll', 'Lu', 'Nd'), whitelist_characters='.-'),
    min_size=1,
    max_size=50
))
@settings(max_examples=200)
def test_is_same_domain_exact_match(host):
    """Property: is_same_domain should return True for exact match"""
    result = is_same_domain(host, host)
    assert result == True, f"Exact match failed: is_same_domain({host!r}, {host!r}) = {result}"

if __name__ == "__main__":
    test_is_same_domain_exact_match()
```

<details>

<summary>
**Failing input**: `host='A'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/40/hypo.py", line 19, in <module>
    test_is_same_domain_exact_match()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/40/hypo.py", line 8, in test_is_same_domain_exact_match
    alphabet=st.characters(whitelist_categories=('Ll', 'Lu', 'Nd'), whitelist_characters='.-'),
               ^^^
  File "/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/40/hypo.py", line 16, in test_is_same_domain_exact_match
    assert result == True, f"Exact match failed: is_same_domain({host!r}, {host!r}) = {result}"
           ^^^^^^^^^^^^^^
AssertionError: Exact match failed: is_same_domain('A', 'A') = False
Falsifying example: test_is_same_domain_exact_match(
    host='A',
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.utils.http import is_same_domain

print("Test 1: is_same_domain('A', 'A')")
print(f"Result: {is_same_domain('A', 'A')}")
print(f"Expected: True\n")

print("Test 2: is_same_domain('a', 'a')")
print(f"Result: {is_same_domain('a', 'a')}")
print(f"Expected: True\n")

print("Test 3: is_same_domain('Example.COM', 'Example.COM')")
print(f"Result: {is_same_domain('Example.COM', 'Example.COM')}")
print(f"Expected: True\n")

print("Test 4: is_same_domain('example.com', 'example.com')")
print(f"Result: {is_same_domain('example.com', 'example.com')}")
print(f"Expected: True\n")

print("Test 5: is_same_domain('Example.com', 'example.com')")
print(f"Result: {is_same_domain('Example.com', 'example.com')}")
print(f"Expected: True (domains are case-insensitive)\n")

print("Test 6: is_same_domain('EXAMPLE.COM', 'example.com')")
print(f"Result: {is_same_domain('EXAMPLE.COM', 'example.com')}")
print(f"Expected: True (domains are case-insensitive)\n")

print("Final assertion test:")
try:
    assert is_same_domain('A', 'A'), "Same domain with same case should match!"
    print("Assertion passed")
except AssertionError as e:
    print(f"AssertionError: {e}")
```

<details>

<summary>
AssertionError: Same domain with same case should match!
</summary>
```
Test 1: is_same_domain('A', 'A')
Result: False
Expected: True

Test 2: is_same_domain('a', 'a')
Result: True
Expected: True

Test 3: is_same_domain('Example.COM', 'Example.COM')
Result: False
Expected: True

Test 4: is_same_domain('example.com', 'example.com')
Result: True
Expected: True

Test 5: is_same_domain('Example.com', 'example.com')
Result: False
Expected: True (domains are case-insensitive)

Test 6: is_same_domain('EXAMPLE.COM', 'example.com')
Result: False
Expected: True (domains are case-insensitive)

Final assertion test:
AssertionError: Same domain with same case should match!
```
</details>

## Why This Is A Bug

This violates RFC 1035 (Domain Names - Implementation and Specification), which explicitly states that all domain name comparisons must be case-insensitive. Specifically:

1. **DNS Standard Violation**: RFC 1035 Section 2.3.3 states: "For all parts of the DNS that are part of the official protocol, all comparisons between character strings (e.g., labels, domain names, etc.) are done in a case-insensitive manner."

2. **Inconsistent Implementation**: The function already lowercases the `pattern` parameter (line 235), indicating the intent for case-insensitive comparison, but fails to lowercase the `host` parameter. This creates an asymmetric comparison where `is_same_domain('EXAMPLE.COM', 'example.com')` returns False but `is_same_domain('example.com', 'EXAMPLE.COM')` returns True.

3. **Function Contract Violation**: The function's docstring promises "exact string match" for non-wildcard patterns. In the context of domain comparison, "exact match" should follow DNS semantics where 'example.com' and 'EXAMPLE.COM' are considered the same domain.

4. **Practical Impact**: This bug can cause CSRF middleware failures when legitimate same-origin requests use mixed-case domain names, potentially blocking valid requests or creating security bypass opportunities if developers work around the issue incorrectly.

## Relevant Context

**Function Location**: `/django/utils/http.py` lines 223-240

**Usage in Django**: This function is used internally by Django's security middleware, particularly in CSRF protection. The `validate_host()` function works around this bug by pre-lowercasing inputs via `split_domain_port()`, but direct usage of `is_same_domain()` (such as in CSRF middleware with URLs from `urlsplit()`) exposes the bug.

**DNS RFC Reference**: [RFC 1035 Section 2.3.3](https://datatracker.ietf.org/doc/html/rfc1035#section-2.3.3)

**Real-world Examples Where This Matters**:
- CDN configurations that preserve original case in Host headers
- Legacy systems that use uppercase domain conventions
- API endpoints that receive user-submitted URLs with mixed casing
- Development environments where developers might use mixed-case hostnames

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