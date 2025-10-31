# Bug Report: starlette.datastructures.URLPath Path Concatenation Without Separator

**Target**: `starlette.datastructures.URLPath.make_absolute_url`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

URLPath.make_absolute_url concatenates paths without a separator when the URLPath doesn't start with "/", producing malformed URLs like `/apitest` instead of `/api/test`.

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

if __name__ == "__main__":
    test_urlpath_handles_relative_paths()
```

<details>

<summary>
**Failing input**: `URLPath('0')` with base_url `"http://example.com/api"`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 21, in <module>
    test_urlpath_handles_relative_paths()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 11, in test_urlpath_handles_relative_paths
    @settings(max_examples=100)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 17, in test_urlpath_handles_relative_paths
    assert not result.path.endswith(relative_path) or result.path.endswith("/" + relative_path), \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Relative path should be separated by '/': got '/api0'
Falsifying example: test_urlpath_handles_relative_paths(
    relative_path='0',
)
```
</details>

## Reproducing the Bug

```python
from starlette.datastructures import URLPath

# Test case 1: URLPath without leading slash
url_path1 = URLPath("test", protocol="", host="")
result1 = url_path1.make_absolute_url("http://example.com/api")
print(f"Test 1: URLPath('test') with base 'http://example.com/api'")
print(f"  Expected: '/api/test'")
print(f"  Got:      '{result1.path}'")
print(f"  Full URL: {result1}")
print()

# Test case 2: URLPath with path segment
url_path2 = URLPath("api/v1", protocol="", host="")
result2 = url_path2.make_absolute_url("http://example.com/api")
print(f"Test 2: URLPath('api/v1') with base 'http://example.com/api'")
print(f"  Expected: '/api/api/v1'")
print(f"  Got:      '{result2.path}'")
print(f"  Full URL: {result2}")
print()

# Test case 3: URLPath with leading slash (normal case)
url_path3 = URLPath("/test", protocol="", host="")
result3 = url_path3.make_absolute_url("http://example.com/api")
print(f"Test 3: URLPath('/test') with base 'http://example.com/api'")
print(f"  Expected: '/api/test'")
print(f"  Got:      '{result3.path}'")
print(f"  Full URL: {result3}")
print()

# Test case 4: Empty URLPath
url_path4 = URLPath("", protocol="", host="")
result4 = url_path4.make_absolute_url("http://example.com/api")
print(f"Test 4: URLPath('') with base 'http://example.com/api'")
print(f"  Expected: '/api'")
print(f"  Got:      '{result4.path}'")
print(f"  Full URL: {result4}")
```

<details>

<summary>
Malformed URL paths produced - paths concatenated without separator
</summary>
```
Test 1: URLPath('test') with base 'http://example.com/api'
  Expected: '/api/test'
  Got:      '/apitest'
  Full URL: http://example.com/apitest

Test 2: URLPath('api/v1') with base 'http://example.com/api'
  Expected: '/api/api/v1'
  Got:      '/apiapi/v1'
  Full URL: http://example.com/apiapi/v1

Test 3: URLPath('/test') with base 'http://example.com/api'
  Expected: '/api/test'
  Got:      '/api/test'
  Full URL: http://example.com/api/test

Test 4: URLPath('') with base 'http://example.com/api'
  Expected: '/api'
  Got:      '/api'
  Full URL: http://example.com/api
```
</details>

## Why This Is A Bug

The bug occurs in the `make_absolute_url` method of the URLPath class at line 196 in `/home/npc/miniconda/lib/python3.13/site-packages/starlette/datastructures.py`:

```python
path = base_url.path.rstrip("/") + str(self)
```

This concatenation logic assumes that `str(self)` (the URLPath value) always starts with a "/" character. However, there is no validation or documentation that enforces this requirement. The URLPath constructor accepts any string value without validation:

```python
def __new__(cls, path: str, protocol: str = "", host: str = "") -> URLPath:
    assert protocol in ("http", "websocket", "")
    return str.__new__(cls, path)  # No validation on path format!
```

This violates expected behavior because:
1. **URLPath is a public API** that users can construct directly - it's not an internal-only class
2. **No validation prevents relative paths** - the constructor accepts any string without checking for leading "/"
3. **No documentation requires absolute paths** - the docstring doesn't state paths must start with "/"
4. **The result is objectively incorrect** - `/apitest` is not a valid concatenation of `/api` and `test`
5. **Standard path joining semantics are violated** - in most path joining operations, joining "/api" and "test" would produce "/api/test"

## Relevant Context

The URLPath class is typically used by Starlette's routing system to generate URLs for named routes. While the routing system always generates paths with leading slashes, URLPath is a public API in `starlette.datastructures` that can be instantiated directly by users.

The bug manifests when users manually create URLPath instances for relative paths, which is a reasonable expectation given:
- The constructor accepts any string
- The class is publicly exported
- There's no documentation stating the path must be absolute

Starlette documentation: https://www.starlette.io/
Source code location: `/home/npc/miniconda/lib/python3.13/site-packages/starlette/datastructures.py:170-198`

## Proposed Fix

The most backwards-compatible fix is to handle relative paths properly in make_absolute_url:

```diff
--- a/starlette/datastructures.py
+++ b/starlette/datastructures.py
@@ -193,7 +193,11 @@ class URLPath(str):
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

This fix:
- Maintains backward compatibility for paths starting with "/"
- Correctly handles relative paths by adding the missing separator
- Handles empty paths correctly
- Follows standard path joining semantics