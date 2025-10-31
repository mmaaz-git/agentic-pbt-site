# Bug Report: GZipMiddleware Case-Sensitive Encoding Check

**Target**: `starlette.middleware.gzip.GZipMiddleware.__call__`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`GZipMiddleware` uses case-sensitive substring matching to check for "gzip" in the Accept-Encoding header, violating HTTP specification that header values are case-insensitive. It also incorrectly matches on substrings rather than tokens.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st


@given(
    case_variant=st.sampled_from(["gzip", "GZIP", "Gzip", "GZip", "gZip", "GzIp"]),
)
@settings(max_examples=50)
def test_gzip_case_insensitive(case_variant):
    current_match = "gzip" in case_variant
    expected_match = True

    assert current_match == expected_match, \
        f"Bug: Accept-Encoding '{case_variant}' should match 'gzip' (HTTP is case-insensitive), but got {current_match}"
```

**Failing input**: `Accept-Encoding: GZIP` (uppercase)

## Reproducing the Bug

```python
from starlette.middleware.gzip import GZipMiddleware

accept_encoding_upper = "GZIP"
accept_encoding_substring = "not-gzip"

current_check_upper = "gzip" in accept_encoding_upper
current_check_substring = "gzip" in accept_encoding_substring

print(f"'gzip' in 'GZIP': {current_check_upper}")
print(f"'gzip' in 'not-gzip': {current_check_substring}")
```

**Output**:
```
'gzip' in 'GZIP': False
'gzip' in 'not-gzip': True
```

At line 24 of `gzip.py`:
```python
if "gzip" in headers.get("Accept-Encoding", ""):
```

This check is:
1. **Case-sensitive**: "GZIP" won't match, even though HTTP headers are case-insensitive per RFC 7231
2. **Substring-based**: "not-gzip" would incorrectly match

## Why This Is A Bug

Per HTTP specifications (RFC 7231 Section 5.3.4), the Accept-Encoding header is case-insensitive. A client sending `Accept-Encoding: GZIP` expects gzip compression to be applied, but the middleware won't apply it due to the case-sensitive check.

Additionally, substring matching can cause false positives. If a hypothetical encoding called "not-gzip" exists, the middleware would incorrectly treat it as requesting gzip compression.

## Fix

```diff
--- a/starlette/middleware/gzip.py
+++ b/starlette/middleware/gzip.py
@@ -21,7 +21,11 @@ class GZipMiddleware:
             return

         headers = Headers(scope=scope)
         responder: ASGIApp
-        if "gzip" in headers.get("Accept-Encoding", ""):
+        accept_encoding = headers.get("Accept-Encoding", "").lower()
+        # Split by comma to get individual encodings, strip whitespace
+        encodings = [e.split(";")[0].strip() for e in accept_encoding.split(",")]
+
+        if "gzip" in encodings:
             responder = GZipResponder(self.app, self.minimum_size, compresslevel=self.compresslevel)
         else:
             responder = IdentityResponder(self.app, self.minimum_size)
```