# Bug Report: django.utils.http.quote_etag Idempotence Violation

**Target**: `django.utils.http.quote_etag`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `quote_etag` function is not idempotent, violating the property implied by its docstring. When called repeatedly on certain inputs (e.g., a single quote character), it continues wrapping the result in additional quotes instead of returning it unchanged.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from django.utils.http import quote_etag

@given(st.text(min_size=1, max_size=100))
@settings(max_examples=500)
def test_quote_etag_idempotent(etag):
    quoted_once = quote_etag(etag)
    quoted_twice = quote_etag(quoted_once)

    assert quoted_once == quoted_twice, \
        f"quote_etag should be idempotent: {etag} -> {quoted_once} -> {quoted_twice}"
```

**Failing input**: `etag='"'`

## Reproducing the Bug

```python
from django.utils.http import quote_etag

etag = '"'
quoted_once = quote_etag(etag)
quoted_twice = quote_etag(quoted_once)

print(f"Original: {etag!r}")
print(f"After 1st quote_etag: {quoted_once!r}")
print(f"After 2nd quote_etag: {quoted_twice!r}")
print(f"Idempotent: {quoted_once == quoted_twice}")
```

Output:
```
Original: '"'
After 1st quote_etag: '"""'
After 2nd quote_etag: '"""""'
Idempotent: False
```

## Why This Is A Bug

The docstring for `quote_etag` states: "If the provided string is already a quoted ETag, return it. Otherwise, wrap the string in quotes, making it a strong ETag."

This implies idempotence: if you call `quote_etag` on a string, and then call it again on the result, you should get the same result. The function is supposed to recognize when a string is already a quoted ETag and return it unchanged.

However, when the input is `'"'`, the function wraps it as `'"""'`. This is not recognized as a valid quoted ETag by the ETAG_MATCH regex (which requires non-quote characters between the quotes), so calling `quote_etag` again wraps it further as `'"""""'`.

## Fix

The function should either:
1. Validate that the input doesn't contain quotes that would create malformed ETags, or
2. Be more lenient in recognizing already-quoted strings

A simple fix would be to check if the string already starts and ends with quotes before wrapping:

```diff
 def quote_etag(etag_str):
     """
     If the provided string is already a quoted ETag, return it. Otherwise, wrap
     the string in quotes, making it a strong ETag.
     """
     if ETAG_MATCH.match(etag_str):
         return etag_str
+    elif etag_str.startswith('"') and etag_str.endswith('"'):
+        return etag_str
     else:
         return '"%s"' % etag_str
```