# Bug Report: Starlette Weak ETag Comparison Fails

**Target**: `starlette.staticfiles.StaticFiles.is_not_modified`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `is_not_modified()` method fails to correctly match weak ETags because it only normalizes the request ETags but not the response ETag, causing valid cache hits to be missed when weak ETags are used.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from starlette.datastructures import Headers

@given(st.text(min_size=1, max_size=20))
def test_etag_weak_and_strong_match(etag_value):
    response_headers = Headers({"etag": f'W/"{etag_value}"'})
    request_headers = Headers({"if-none-match": f'W/"{etag_value}"'})

    from starlette.staticfiles import StaticFiles
    static_files = StaticFiles(directory="/tmp")

    result = static_files.is_not_modified(response_headers, request_headers)
    assert result is True, f"Weak ETag W/\"{etag_value}\" should match itself"
```

**Failing input**: Any weak ETag like `W/"123"`

## Reproducing the Bug

```python
from starlette.staticfiles import StaticFiles
from starlette.datastructures import Headers

static_files = StaticFiles(directory="/tmp", check_dir=False)

response_headers = Headers({"etag": 'W/"123"'})
request_headers = Headers({"if-none-match": 'W/"123"'})

result = static_files.is_not_modified(response_headers, request_headers)
print(f"Weak ETag match: {result}")

response_headers2 = Headers({"etag": '"456"'})
request_headers2 = Headers({"if-none-match": '"456"'})

result2 = static_files.is_not_modified(response_headers2, request_headers2)
print(f"Strong ETag match: {result2}")
```

Output:
```
Weak ETag match: False  # BUG: Should be True!
Strong ETag match: True
```

## Why This Is A Bug

The bug is in line 207 of `starlette/staticfiles.py`:

```python
if etag in [tag.strip(" W/") for tag in if_none_match.split(",")]:
    return True
```

The code:
1. Splits the request's `if-none-match` header by comma
2. Strips ` `, `W`, and `/` characters from each tag
3. Compares the **unmodified** response `etag` against these stripped values

For example, with `etag = 'W/"123"'` and `if_none_match = 'W/"123"'`:
1. `if_none_match.split(",")` → `['W/"123"']`
2. `tag.strip(" W/")` → `'"123"'` (W and / stripped)
3. `'W/"123"' in ['"123"']` → `False` (mismatch!)

The response ETag `'W/"123"'` is never in the list `['"123"']` because it wasn't normalized.

Additionally, `.strip(" W/")` has another subtle bug: it strips *characters*, not the *string* "W/". This means:
- `'W/"abc"//WW'.strip(" W/")` → `'"abc"'` (strips all W, /, and spaces from both ends)
- This could cause false matches in edge cases

## Fix

```diff
     def is_not_modified(self, response_headers: Headers, request_headers: Headers) -> bool:
         """
         Given the request and response headers, return `True` if an HTTP
         "Not Modified" response could be returned instead.
         """
         try:
             if_none_match = request_headers["if-none-match"]
             etag = response_headers["etag"]
-            if etag in [tag.strip(" W/") for tag in if_none_match.split(",")]:
+            # Normalize both weak and strong ETags for comparison
+            # Weak ETags start with 'W/' prefix
+            response_etag_normalized = etag.strip().removeprefix("W/").strip()
+            request_etags_normalized = [
+                tag.strip().removeprefix("W/").strip()
+                for tag in if_none_match.split(",")
+            ]
+            if response_etag_normalized in request_etags_normalized:
                 return True
         except KeyError:
             pass

         try:
             if_modified_since = parsedate(request_headers["if-modified-since"])
             last_modified = parsedate(response_headers["last-modified"])
             if if_modified_since is not None and last_modified is not None and if_modified_since >= last_modified:
                 return True
         except KeyError:
             pass

         return False
```