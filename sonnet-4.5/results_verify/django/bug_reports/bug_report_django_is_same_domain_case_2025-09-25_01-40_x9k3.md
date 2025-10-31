# Bug Report: django.utils.http.is_same_domain Case Sensitivity Bug

**Target**: `django.utils.http.is_same_domain`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `is_same_domain` function performs case-sensitive comparison when checking if a host matches a pattern, even though DNS is case-insensitive. The function lowercases the `pattern` parameter but not the `host` parameter, causing identical domains with different casing to be considered non-matching.

## Property-Based Test

```python
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
```

**Failing input**: `host='A'`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.utils.http import is_same_domain

print(is_same_domain('A', 'A'))            # False (should be True)
print(is_same_domain('a', 'a'))            # True
print(is_same_domain('Example.COM', 'Example.COM'))  # False (should be True)
print(is_same_domain('example.com', 'example.com'))  # True

assert is_same_domain('A', 'A'), "Same domain with same case should match!"
```

Output:
```
False
True
False
True
AssertionError: Same domain with same case should match!
```

## Why This Is A Bug

DNS (Domain Name System) is case-insensitive according to RFC 1035. This means:
- `Example.com`, `EXAMPLE.COM`, and `example.com` all refer to the same domain
- Any domain validation function should treat these as identical
- The function's docstring says "exact match" which implies case-insensitive for domains

The bug occurs because:
1. Line 235 lowercases only `pattern`: `pattern = pattern.lower()`
2. Line 239 compares: `pattern == host`
3. If `host` has uppercase letters, `pattern.lower() == host` fails even when they're the same domain

This breaks the documented behavior of "exact string match" for domains, since domains should match case-insensitively.

## Fix

Lowercase both `pattern` and `host` before comparison:

```diff
--- a/django/utils/http.py
+++ b/django/utils/http.py
@@ -232,9 +232,10 @@ def is_same_domain(host, pattern):
     if not pattern:
         return False

+    host = host.lower()
     pattern = pattern.lower()
     return (
         pattern[0] == "."
         and (host.endswith(pattern) or host == pattern[1:])
         or pattern == host
     )
```

This ensures both values are lowercased before any comparison, making the function properly case-insensitive.