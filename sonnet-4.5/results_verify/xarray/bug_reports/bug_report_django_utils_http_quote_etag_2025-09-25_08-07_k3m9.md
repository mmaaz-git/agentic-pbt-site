# Bug Report: django.utils.http.quote_etag Non-Idempotence

**Target**: `django.utils.http.quote_etag`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `quote_etag` function violates its documented idempotence property when the input contains quote characters. The docstring states "If the provided string is already a quoted ETag, return it," implying that `quote_etag(quote_etag(x)) == quote_etag(x)`. However, for inputs containing quotes, repeated calls produce different results.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.utils.http import quote_etag

@given(st.text(min_size=1, max_size=100))
def test_quote_etag_idempotent(etag):
    quoted_once = quote_etag(etag)
    quoted_twice = quote_etag(quoted_once)
    assert quoted_once == quoted_twice, \
        f"quote_etag should be idempotent: {etag} -> {quoted_once} -> {quoted_twice}"
```

**Failing input**: `'"'`

## Reproducing the Bug

```python
from django.utils.http import quote_etag

etag = '"'

result1 = quote_etag(etag)
print(f"First call:  {result1!r}")

result2 = quote_etag(result1)
print(f"Second call: {result2!r}")

result3 = quote_etag(result2)
print(f"Third call:  {result3!r}")

assert result1 == result2, f"Expected idempotence, got {result1!r} != {result2!r}"
```

**Output:**
```
First call:  '"""'
Second call: '"""""'
Third call:  '"""""""'
AssertionError: Expected idempotence, got '"""' != '"""""'
```

## Why This Is A Bug

1. **Contract Violation**: The docstring explicitly states "If the provided string is already a quoted ETag, return it," which implies idempotence.

2. **Unbounded Growth**: Repeated applications cause the string to grow without bound, which could lead to issues in long-running processes or retry logic.

3. **Silent Failure**: The function silently produces incorrect results instead of either handling the input gracefully or raising an error for invalid input.

4. **Regex Mismatch**: The `ETAG_MATCH` regex pattern expects `[^"]*` (no quotes inside), but `quote_etag` doesn't validate this before wrapping, leading to inconsistent behavior.

While RFC 9110-compliant ETags don't contain quote characters, defensive programming would suggest the function should either:
- Be idempotent for all inputs (including edge cases), OR
- Raise an error for invalid ETags

## Fix

The fix should make `quote_etag` truly idempotent by checking if wrapping the input would create a valid quoted ETag:

```diff
--- a/django/utils/http.py
+++ b/django/utils/http.py
@@ -212,10 +212,16 @@ def quote_etag(etag_str):
 def quote_etag(etag_str):
     """
     If the provided string is already a quoted ETag, return it. Otherwise, wrap
     the string in quotes, making it a strong ETag.
     """
     if ETAG_MATCH.match(etag_str):
         return etag_str
     else:
-        return '"%s"' % etag_str
+        wrapped = '"%s"' % etag_str
+        # Ensure idempotence: if wrapping creates a valid ETag, return it
+        # Otherwise, escape quotes in the input to maintain RFC compliance
+        if ETAG_MATCH.match(wrapped):
+            return wrapped
+        else:
+            # Input contains quotes; escape them or handle as error
+            raise ValueError(f"Invalid ETag value containing quotes: {etag_str!r}")
```

Alternatively, for backward compatibility, simply escape quotes in the input:

```diff
--- a/django/utils/http.py
+++ b/django/utils/http.py
@@ -212,10 +212,11 @@ def quote_etag(etag_str):
 def quote_etag(etag_str):
     """
     If the provided string is already a quoted ETag, return it. Otherwise, wrap
     the string in quotes, making it a strong ETag.
     """
     if ETAG_MATCH.match(etag_str):
         return etag_str
     else:
+        # Remove any quotes from the input to ensure RFC compliance
+        etag_str = etag_str.replace('"', '')
         return '"%s"' % etag_str
```