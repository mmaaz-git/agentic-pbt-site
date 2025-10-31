# Bug Report: starlette.staticfiles ETag Parsing Corrupts ETags Ending with 'W' or '/'

**Target**: `starlette.staticfiles.StaticFiles.is_not_modified`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `is_not_modified` method in `StaticFiles` uses `strip(" W/")` to parse ETags from the `If-None-Match` header, which incorrectly strips characters from both ends of the string. This corrupts ETags that end with 'W' or '/' characters, causing false negatives in cache validation.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/starlette_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from starlette.staticfiles import StaticFiles
from starlette.datastructures import Headers


@given(st.text(alphabet=st.characters(min_codepoint=33, max_codepoint=126, blacklist_characters='"'), min_size=1))
@settings(max_examples=1000)
def test_etag_round_trip_strong(etag_value):
    """
    Property: If response has a strong ETag, and request sends the same ETag
    in if-none-match, is_not_modified should return True.
    """
    sf = StaticFiles(directory=".", check_dir=False)

    etag = f'"{etag_value}"'

    response_headers = Headers({"etag": etag})
    request_headers = Headers({"if-none-match": etag})

    result = sf.is_not_modified(response_headers, request_headers)

    assert result, f"Expected is_not_modified to return True for matching ETag {etag}, but got False"
```

**Failing input**: Any ETag ending with 'W' or '/', such as `"abcW"`, `"test/"`, `"dataW/"`, etc.

## Reproducing the Bug

```python
from starlette.staticfiles import StaticFiles
from starlette.datastructures import Headers

sf = StaticFiles(directory=".", check_dir=False)

response_etag = '"abcW"'
response_headers = Headers({"etag": response_etag})
request_headers = Headers({"if-none-match": response_etag})

result = sf.is_not_modified(response_headers, request_headers)

print(f"Result: {result}")

tags = [tag.strip(" W/") for tag in response_etag.split(",")]
print(f"Parsed: {tags}")
```

**Expected output**: `Result: True`, `Parsed: ['"abcW"']`
**Actual output**: `Result: False`, `Parsed: ['"abc"']` (W stripped from end)

## Why This Is A Bug

The code on line 207 of `starlette/staticfiles.py` uses:

```python
if etag in [tag.strip(" W/") for tag in if_none_match.split(",")]
```

Python's `str.strip()` removes characters from **both ends** of the string, not just the beginning. This causes:

1. `W/"abc"` → `"abc"` ✓ (correct: weak ETag prefix removed)
2. `"abcW"` → `"abc"` ✗ (incorrect: trailing W removed, missing closing quote)
3. `"abc/"` → `"abc"` ✗ (incorrect: trailing / removed, missing closing quote)

According to RFC 7232, ETags are opaque quoted strings that can contain any characters. The current implementation corrupts ETags that legitimately end with 'W' or '/' characters, causing cache misses when the client and server agree on the resource version.

## Fix

```diff
--- a/starlette/staticfiles.py
+++ b/starlette/staticfiles.py
@@ -204,7 +204,10 @@ class StaticFiles:
         try:
             if_none_match = request_headers["if-none-match"]
             etag = response_headers["etag"]
-            if etag in [tag.strip(" W/") for tag in if_none_match.split(",")]:
+            # Parse ETags, handling weak ETags (W/"...") by removing only the W/ prefix
+            parsed_tags = []
+            for tag in if_none_match.split(","):
+                tag = tag.strip()
+                if tag.startswith("W/"):
+                    tag = tag[2:]
+                parsed_tags.append(tag)
+            if etag in parsed_tags:
                 return True
         except KeyError:
             pass
```

Alternative one-liner fix using `removeprefix` (Python 3.9+):

```diff
--- a/starlette/staticfiles.py
+++ b/starlette/staticfiles.py
@@ -204,7 +204,7 @@ class StaticFiles:
         try:
             if_none_match = request_headers["if-none-match"]
             etag = response_headers["etag"]
-            if etag in [tag.strip(" W/") for tag in if_none_match.split(",")]:
+            if etag in [tag.strip().removeprefix("W/") for tag in if_none_match.split(",")]:
                 return True
         except KeyError:
             pass
```