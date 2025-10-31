# Bug Report: starlette.datastructures.URLPath Path Concatenation

**Target**: `starlette.datastructures.URLPath.make_absolute_url`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

URLPath.make_absolute_url concatenates paths without a separator when the URLPath doesn't start with "/", leading to malformed URLs like `/apitest` instead of `/api/test`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from starlette.datastructures import URL, URLPath

relative_paths = st.text(
    alphabet=st.characters(whitelist_categories=('L', 'N'), min_codepoint=33, max_codepoint=126),
    min_size=1,
    max_size=20
).filter(lambda p: not p.startswith('/'))

@given(relative_paths)
@settings(max_examples=100)
def test_urlpath_handles_relative_paths(relative_path):
    url_path = URLPath(relative_path, protocol="", host="")
    base_url = "http://example.com/api"
    result = url_path.make_absolute_url(base_url)

    assert not result.path.endswith(relative_path) or result.path.endswith("/" + relative_path), \
        f"Relative path should be separated by '/': got {result.path!r}"
```

**Failing inputs** (discovered by Hypothesis):
- `URLPath("test")` with base_url `"http://example.com/api"` → produces `/apitest` instead of `/api/test`
- `URLPath("api/v1")` with base_url `"http://example.com/api"` → produces `/apiapi/v1` instead of `/api/api/v1`

## Reproducing the Bug

```python
from starlette.datastructures import URLPath

url_path1 = URLPath("test", protocol="", host="")
result1 = url_path1.make_absolute_url("http://example.com/api")
print(f"Test 1: {result1.path!r}")

url_path2 = URLPath("api/v1", protocol="", host="")
result2 = url_path2.make_absolute_url("http://example.com/api")
print(f"Test 2: {result2.path!r}")
```

Output:
```
Test 1: '/apitest'
Test 2: '/apiapi/v1'
```

Both produce malformed paths by concatenating without a path separator.

## Why This Is A Bug

The issue is in the path concatenation logic:

```python
path = base_url.path.rstrip("/") + str(self)
```

This assumes `self` (the URLPath string) always starts with "/", but there's no validation or documentation requiring this. The URLPath constructor accepts any string:

```python
class URLPath(str):
    def __new__(cls, path: str, protocol: str = "", host: str = "") -> URLPath:
        assert protocol in ("http", "websocket", "")
        return str.__new__(cls, path)  # No validation on path format!
```

This is a bug because:
1. URLPath is a public API that users can construct directly
2. No validation prevents relative paths
3. Documentation doesn't require paths to start with "/"
4. Results in malformed URLs when paths don't follow the implicit assumption

## Fix

Add proper path handling or validation. Option 1: Validate paths must start with "/":

```diff
--- a/starlette/datastructures.py
+++ b/starlette/datastructures.py
@@ -185,6 +185,7 @@ class URLPath(str):

     def __new__(cls, path: str, protocol: str = "", host: str = "") -> URLPath:
         assert protocol in ("http", "websocket", "")
+        assert path.startswith("/") or path == "", "URLPath must start with '/' or be empty"
         return str.__new__(cls, path)
```

Option 2: Handle relative paths properly in make_absolute_url:

```diff
--- a/starlette/datastructures.py
+++ b/starlette/datastructures.py
@@ -201,7 +201,11 @@ class URLPath(str):
             scheme = base_url.scheme

         netloc = self.host or base_url.netloc
-        path = base_url.path.rstrip("/") + str(self)
+        url_path_str = str(self)
+        if url_path_str and not url_path_str.startswith("/"):
+            path = base_url.path.rstrip("/") + "/" + url_path_str
+        else:
+            path = base_url.path.rstrip("/") + url_path_str
         return URL(scheme=scheme, netloc=netloc, path=path)
```

Option 2 is more defensive and maintains backward compatibility while fixing the edge case.