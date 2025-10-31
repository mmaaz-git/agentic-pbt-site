# Bug Report: django.utils.http.quote_etag Violates Idempotence Property

**Target**: `django.utils.http.quote_etag`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `quote_etag` function violates its documented idempotence property when given input strings containing quote characters. The docstring states "If the provided string is already a quoted ETag, return it", but calling the function twice on certain inputs produces different results.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.utils.http import quote_etag


@given(st.text())
def test_quote_etag_idempotence(s):
    result1 = quote_etag(s)
    result2 = quote_etag(result1)
    assert result1 == result2, f"quote_etag is not idempotent: quote_etag({s!r}) = {result1!r}, but quote_etag({result1!r}) = {result2!r}"
```

**Failing input**: `'"'` (a single quote character)

## Reproducing the Bug

```python
from django.utils.http import quote_etag

s = '"'
result1 = quote_etag(s)
result2 = quote_etag(result1)

print(f"quote_etag({s!r}) = {result1!r}")
print(f"quote_etag({result1!r}) = {result2!r}")

assert result1 == result2
```

Output:
```
quote_etag('"') = '"""'
quote_etag('"""') = '"""""'
AssertionError
```

## Why This Is A Bug

The function's docstring explicitly promises: "If the provided string is already a quoted ETag, return it. Otherwise, wrap the string in quotes, making it a strong ETag."

This implies idempotence: calling `quote_etag` on its own output should return the same value. However, the function fails this property when the input contains quote characters.

The root cause is that the ETAG_MATCH regex pattern `"[^"]*"` only matches quoted strings that don't contain internal quotes. When `quote_etag` wraps a string containing quotes (like `"`), the result (`"""`) doesn't match the pattern, so a subsequent call wraps it again.

This violates the API contract and could cause issues in production where `quote_etag` might be called multiple times on the same ETag value (e.g., through middleware or caching layers).

## Fix

The function should either:
1. Escape quote characters in the input before wrapping, or
2. Detect when the output doesn't match the ETAG pattern and handle it appropriately

A simple fix using approach 1:

```diff
def quote_etag(etag_str):
    """
    If the provided string is already a quoted ETag, return it. Otherwise, wrap
    the string in quotes, making it a strong ETag.
    """
    if ETAG_MATCH.match(etag_str):
        return etag_str
    else:
-       return '"%s"' % etag_str
+       escaped = etag_str.replace('\\', '\\\\').replace('"', '\\"')
+       return '"%s"' % escaped
```

However, this changes the behavior for existing code. A safer fix might be to check if wrapping produces a valid ETag, and if not, escape the input:

```diff
def quote_etag(etag_str):
    """
    If the provided string is already a quoted ETag, return it. Otherwise, wrap
    the string in quotes, making it a strong ETag.
    """
    if ETAG_MATCH.match(etag_str):
        return etag_str
    else:
-       return '"%s"' % etag_str
+       # If the string contains quotes, we need to escape them
+       if '"' in etag_str:
+           escaped = etag_str.replace('\\', '\\\\').replace('"', '\\"')
+           return '"%s"' % escaped
+       else:
+           return '"%s"' % etag_str
```