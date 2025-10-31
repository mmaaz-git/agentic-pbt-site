# Bug Report: starlette.staticfiles Weak ETag Matching

**Target**: `starlette.staticfiles.StaticFiles.is_not_modified`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `is_not_modified` method in `StaticFiles` incorrectly handles weak ETag matching when the ETag value ends with characters 'W', '/', or space. The bug uses `strip(" W/")` which removes these characters from both ends of the string, when it should only remove the "W/" prefix.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/starlette_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, example, settings
from starlette.datastructures import Headers
from starlette.staticfiles import StaticFiles


@settings(max_examples=1000)
@given(
    st.text(
        alphabet=st.characters(
            blacklist_categories=('Cs', 'Cc'),
            blacklist_characters=(',', '\r', '\n')
        ),
        min_size=1
    )
)
@example("W")
@example("/")
@example("W/")
@example("testW")
@example("test/")
def test_weak_etag_matching_property(etag_value):
    sf = StaticFiles(directory="/tmp", check_dir=False)

    strong_etag = f'"{etag_value}"'
    weak_etag = f'W/{strong_etag}'

    response_headers = Headers({"etag": strong_etag})
    request_headers = Headers({"if-none-match": weak_etag})

    result = sf.is_not_modified(response_headers, request_headers)

    assert result, f"Weak ETag {weak_etag} should match strong ETag {strong_etag}, but doesn't"
```

**Failing input**: `etag_value="testW"`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/starlette_env/lib/python3.13/site-packages')

from starlette.datastructures import Headers
from starlette.staticfiles import StaticFiles

sf = StaticFiles(directory="/tmp", check_dir=False)

strong_etag = '"testW"'
weak_etag = 'W/"testW"'

response_headers = Headers({"etag": strong_etag})
request_headers = Headers({"if-none-match": weak_etag})

result = sf.is_not_modified(response_headers, request_headers)

print(f"Strong ETag: {strong_etag}")
print(f"Weak ETag: {weak_etag}")
print(f"Matches: {result}")
print(f"Expected: True")
print(f"Actual: {result}")

if_none_match = weak_etag
tags = [tag.strip(" W/") for tag in if_none_match.split(",")]
print(f"\nAfter strip(' W/'): {tags}")
print(f"Expected: ['\"testW\"']")
print(f"Problem: strip() removes W from BOTH ends!")
```

## Why This Is A Bug

According to RFC 7232, weak ETags are prefixed with "W/" and should match their strong counterparts for comparison purposes in conditional requests. The current implementation uses `strip(" W/")` which removes any combination of 'W', '/', and space characters from both ends of the string.

This causes false negatives when:
- ETag values end with 'W' (e.g., `"testW"`)
- ETag values end with '/' (e.g., `"test/"`)
- ETag values end with space (less common but possible)

For example, weak ETag `W/"testW"` after `strip(" W/")` becomes `"test` instead of `"testW"`, failing to match the strong ETag `"testW"`.

## Fix

```diff
--- a/starlette/staticfiles.py
+++ b/starlette/staticfiles.py
@@ -204,7 +204,10 @@ class StaticFiles:
         try:
             if_none_match = request_headers["if-none-match"]
             etag = response_headers["etag"]
-            if etag in [tag.strip(" W/") for tag in if_none_match.split(",")]:
+            if etag in [
+                tag.strip().removeprefix("W/") if tag.strip().startswith("W/")
+                else tag.strip() for tag in if_none_match.split(",")
+            ]:
                 return True
         except KeyError:
             pass
```