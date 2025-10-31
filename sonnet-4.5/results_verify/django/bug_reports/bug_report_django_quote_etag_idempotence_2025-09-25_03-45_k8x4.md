# Bug Report: django.utils.http.quote_etag Idempotence Violation

**Target**: `django.utils.http.quote_etag`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `quote_etag` function violates its documented idempotence property. When given an input containing quote characters, repeatedly calling `quote_etag` on its output causes unbounded string growth instead of returning the same value.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings as hypothesis_settings
from django.utils.http import quote_etag

@given(st.text(min_size=1, max_size=100))
@hypothesis_settings(max_examples=1000)
def test_quote_etag_idempotence(etag):
    quoted_once = quote_etag(etag)
    quoted_twice = quote_etag(quoted_once)
    assert quoted_once == quoted_twice, \
        f"quote_etag not idempotent: {etag!r} -> {quoted_once!r} -> {quoted_twice!r}"
```

**Failing input**: `'"'` (single quote character)

## Reproducing the Bug

```python
from django.utils.http import quote_etag

etag = '"'
result1 = quote_etag(etag)
result2 = quote_etag(result1)
result3 = quote_etag(result2)

print(f"Input:  {etag!r}")
print(f"Once:   {result1!r}")
print(f"Twice:  {result2!r}")
print(f"Thrice: {result3!r}")
```

Output:
```
Input:  '"'
Once:   '"""'
Twice:  '"""""'
Thrice: '"""""""'
```

## Why This Is A Bug

The function's docstring states: "If the provided string is already a quoted ETag, return it." This clearly implies idempotence: `quote_etag(quote_etag(x)) == quote_etag(x)`.

However, the ETAG_MATCH regex requires that the content between quotes contains only non-quote characters (`[^"]*`). When the input is `'"'`:
1. It doesn't match the regex (single quote, not a pair)
2. Gets wrapped: `'"""'` (three quotes)
3. Still doesn't match (three quotes, regex expects exactly two with non-quote content)
4. Gets wrapped again: `'"""""'` (five quotes)
5. Process continues indefinitely

This violates the documented idempotence property and can cause issues if `quote_etag` is called multiple times on the same value in loops or recursive contexts.

## Fix

The regex should allow quote characters within the ETag content. According to RFC 9110, ETags can contain any characters except the delimiting quotes. The fix is to change `[^"]*` to `.*?` (non-greedy match):

```diff
--- a/django/utils/http.py
+++ b/django/utils/http.py
@@ -17,7 +17,7 @@ ETAG_MATCH = _lazy_re_compile(
     \A(      # start of string and capture group
     (?:W/)?  # optional weak indicator
     "        # opening quote
-    [^"]*    # any sequence of non-quote characters
+    .*?      # any sequence of characters (non-greedy)
     "        # end quote
     )\Z      # end of string and capture group
 """,
```

Wait, this won't work either because `.*?` would match the first quote it encounters. The actual issue is that RFC 9110 defines ETag values as quoted strings where internal quotes must be escaped. However, Django's `quote_etag` doesn't handle escaping.

A better fix is to make `quote_etag` truly idempotent by checking if the string already starts and ends with quotes:

```diff
--- a/django/utils/http.py
+++ b/django/utils/http.py
@@ -212,10 +212,12 @@ def parse_etags(etag_str):
 def quote_etag(etag_str):
     """
     If the provided string is already a quoted ETag, return it. Otherwise, wrap
     the string in quotes, making it a strong ETag.
     """
-    if ETAG_MATCH.match(etag_str):
+    # Check if already quoted (strong or weak ETag)
+    if (etag_str.startswith('"') and etag_str.endswith('"')) or \
+       (etag_str.startswith('W/"') and etag_str.endswith('"')):
         return etag_str
     else:
         return '"%s"' % etag_str
```

This makes the function idempotent by using a simple heuristic: if the string already starts and ends with quotes (or is a weak ETag), return it unchanged.