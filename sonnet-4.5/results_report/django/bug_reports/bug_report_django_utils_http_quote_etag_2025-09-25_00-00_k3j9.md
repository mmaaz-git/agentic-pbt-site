# Bug Report: django.utils.http.quote_etag Idempotence Violation

**Target**: `django.utils.http.quote_etag`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `quote_etag` function violates its documented contract of idempotence. When passed a string containing only a double-quote character (`"`), it keeps wrapping it in additional quotes on each call instead of recognizing it as an already-quoted ETag.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.utils.http import quote_etag

@given(st.text(min_size=1, max_size=100))
def test_quote_etag_idempotent(etag):
    quoted_once = quote_etag(etag)
    quoted_twice = quote_etag(quoted_once)
    assert quoted_once == quoted_twice
```

**Failing input**: `'"'`

## Reproducing the Bug

```python
from django.utils.http import quote_etag

etag = '"'
print(f"Input:        {etag!r}")
print(f"After 1 call: {quote_etag(etag)!r}")
print(f"After 2 calls: {quote_etag(quote_etag(etag))!r}")
print(f"After 3 calls: {quote_etag(quote_etag(quote_etag(etag)))!r}")
```

Output:
```
Input:        '"'
After 1 call: '"""'
After 2 calls: '"""""'
After 3 calls: '"""""""'
```

## Why This Is A Bug

The function's docstring explicitly states: "If the provided string is already a quoted ETag, return it." This implies the function should be idempotent - calling it multiple times should produce the same result as calling it once.

The issue is in the `ETAG_MATCH` regex pattern at line 15-25 of `django/utils/http.py`:

```python
ETAG_MATCH = _lazy_re_compile(
    r"""
    \A(      # start of string and capture group
    (?:W/)?  # optional weak indicator
    "        # opening quote
    [^"]*    # any sequence of non-quote characters  <-- PROBLEM
    "        # end quote
    )\Z      # end of string and capture group
""",
    re.X,
)
```

The pattern `[^"]*` requires zero or more **non-quote** characters between the quotes. This means:
- `'"""'` (the result of quoting `"`) does NOT match because it has a `"` between the outer quotes
- So it gets wrapped again to `'"""""'`
- Which also doesn't match, leading to `'"""""""'`, and so on

## Fix

The regex should allow quote characters to be present between the outer quotes, since ETags can contain any characters except unescaped quotes. RFC 9110 specifies that ETags are opaque quoted strings.

```diff
--- a/django/utils/http.py
+++ b/django/utils/http.py
@@ -18,7 +18,7 @@ ETAG_MATCH = _lazy_re_compile(
     \A(      # start of string and capture group
     (?:W/)?  # optional weak indicator
     "        # opening quote
-    [^"]*    # any sequence of non-quote characters
+    .*?      # any sequence of characters (non-greedy)
     "        # end quote
     )\Z      # end of string and capture group
 """,
```

This change makes the pattern match any characters between the quotes (non-greedy), which correctly identifies already-quoted ETags like `'"""'` while still matching standard ETags.