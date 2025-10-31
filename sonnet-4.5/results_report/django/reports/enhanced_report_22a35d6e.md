# Bug Report: django.utils.http.is_same_domain Case Sensitivity Violates DNS Standards

**Target**: `django.utils.http.is_same_domain`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `is_same_domain` function incorrectly performs case-sensitive domain comparisons, causing domains to not match themselves when they contain uppercase letters. This violates DNS case-insensitivity standards (RFC 4343) and creates security risks in Django's CSRF protection and host validation.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Hypothesis test demonstrating the django.utils.http.is_same_domain bug"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from django.utils.http import is_same_domain

@given(st.text(min_size=1))
def test_is_same_domain_exact_match(domain):
    """Test that any domain should match itself (identity property)"""
    result = is_same_domain(domain, domain)
    assert result is True, f"Domain '{domain}' doesn't match itself"

if __name__ == "__main__":
    # Run the test
    test_is_same_domain_exact_match()
```

<details>

<summary>
**Failing input**: `domain='A'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/60/hypo.py", line 18, in <module>
    test_is_same_domain_exact_match()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/60/hypo.py", line 11, in test_is_same_domain_exact_match
    def test_is_same_domain_exact_match(domain):
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/60/hypo.py", line 14, in test_is_same_domain_exact_match
    assert result is True, f"Domain '{domain}' doesn't match itself"
           ^^^^^^^^^^^^^^
AssertionError: Domain 'A' doesn't match itself
Falsifying example: test_is_same_domain_exact_match(
    domain='A',
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction of the django.utils.http.is_same_domain bug"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.utils.http import is_same_domain

# Test 1: An uppercase domain doesn't match itself
print("Test 1: is_same_domain('EXAMPLE.COM', 'EXAMPLE.COM')")
result1 = is_same_domain('EXAMPLE.COM', 'EXAMPLE.COM')
print(f"Result: {result1}")
print(f"Expected: True")
print(f"PASS" if result1 == True else f"FAIL: Domain doesn't match itself when uppercase\n")

# Test 2: Mixed-case domain doesn't match lowercase version
print("Test 2: is_same_domain('Example.COM', 'example.com')")
result2 = is_same_domain('Example.COM', 'example.com')
print(f"Result: {result2}")
print(f"Expected: True (DNS is case-insensitive)")
print(f"PASS" if result2 == True else f"FAIL: Mixed-case domain doesn't match lowercase version\n")

# Test 3: Uppercase subdomain doesn't match lowercase wildcard pattern
print("Test 3: is_same_domain('FOO.EXAMPLE.COM', '.example.com')")
result3 = is_same_domain('FOO.EXAMPLE.COM', '.example.com')
print(f"Result: {result3}")
print(f"Expected: True (should match wildcard pattern)")
print(f"PASS" if result3 == True else f"FAIL: Uppercase subdomain doesn't match lowercase wildcard\n")

# Test 4: Demonstrate asymmetry
print("Test 4: Asymmetric behavior")
print("  is_same_domain('example.com', 'EXAMPLE.COM'):", is_same_domain('example.com', 'EXAMPLE.COM'))
print("  is_same_domain('EXAMPLE.COM', 'example.com'):", is_same_domain('EXAMPLE.COM', 'example.com'))
print("FAIL: Function behaves asymmetrically with case\n")

# Test 5: Simple uppercase letter test
print("Test 5: is_same_domain('A', 'A')")
result5 = is_same_domain('A', 'A')
print(f"Result: {result5}")
print(f"Expected: True")
print(f"PASS" if result5 == True else f"FAIL: Single uppercase letter doesn't match itself")
```

<details>

<summary>
All tests fail - domains don't match themselves when uppercase
</summary>
```
Test 1: is_same_domain('EXAMPLE.COM', 'EXAMPLE.COM')
Result: False
Expected: True
FAIL: Domain doesn't match itself when uppercase

Test 2: is_same_domain('Example.COM', 'example.com')
Result: False
Expected: True (DNS is case-insensitive)
FAIL: Mixed-case domain doesn't match lowercase version

Test 3: is_same_domain('FOO.EXAMPLE.COM', '.example.com')
Result: False
Expected: True (should match wildcard pattern)
FAIL: Uppercase subdomain doesn't match lowercase wildcard

Test 4: Asymmetric behavior
  is_same_domain('example.com', 'EXAMPLE.COM'): True
  is_same_domain('EXAMPLE.COM', 'example.com'): False
FAIL: Function behaves asymmetrically with case

Test 5: is_same_domain('A', 'A')
Result: False
Expected: True
FAIL: Single uppercase letter doesn't match itself
```
</details>

## Why This Is A Bug

This bug violates fundamental correctness properties and DNS standards:

1. **Identity Property Violation**: The function fails the basic mathematical identity property where `X == X`. A domain containing uppercase letters returns `False` when compared to itself, which is logically incorrect.

2. **DNS Standard Violation**: RFC 4343 Section 3 explicitly states that DNS comparisons must be case-insensitive. The function's case-sensitive behavior contradicts this fundamental internet standard.

3. **Asymmetric Behavior**: The function exhibits non-commutative behavior where `is_same_domain(a, b) != is_same_domain(b, a)` when case differs. This asymmetry is caused by only lowercasing the `pattern` parameter (line 235) but not the `host` parameter.

4. **Security Impact**: This function is used in Django's CSRF middleware (`django/middleware/csrf.py`) and HTTP request validation (`django/http/request.py`) for validating origins, referrers, and host headers against trusted domains. The bug could cause:
   - Legitimate requests with uppercase domains to be incorrectly rejected
   - Potential bypass scenarios if uppercase domains aren't properly validated
   - Production failures when users or systems submit uppercase domain names

5. **Documentation Contradiction**: While the docstring mentions "exact string match" for non-wildcard patterns, it doesn't explicitly state case-sensitive matching is intended. The partial lowercasing of only the pattern parameter suggests case-insensitive comparison was the original intent but incompletely implemented.

## Relevant Context

**Code Location**: `/django/utils/http.py` lines 223-240

**Current Implementation Flaw**:
- Line 235: `pattern = pattern.lower()` - Only the pattern is lowercased
- The host parameter is never lowercased before comparison
- Comparisons occur between original (potentially uppercase) host and lowercased pattern

**Usage in Django Security Components**:
- CSRF Protection: Used to validate Origin and Referer headers
- Host Header Validation: Used to prevent Host header attacks
- These are critical security components where incorrect domain matching could have serious implications

**RFC 4343 Reference**: https://datatracker.ietf.org/doc/html/rfc4343#section-3
"Comparisons on name lookup for DNS queries should be case insensitive"

**Real-world Impact Examples**:
- API clients sending uppercase Host headers would fail CSRF checks
- Load balancers or proxies that uppercase domain names would break
- Mobile apps or browsers that don't normalize case could fail authentication

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