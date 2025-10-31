# Bug Report: pydantic.networks.AnyUrl.build() Duplicates Root Path

**Target**: `pydantic.networks.AnyUrl.build()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`AnyUrl.build()` incorrectly duplicates the root path `/` to `//` when `path='/'` is explicitly provided. This violates the round-trip property, as the extracted `.path` property returns `//` instead of `/`.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
from pydantic.networks import AnyUrl

@st.composite
def url_build_components(draw):
    scheme = draw(st.sampled_from(['http', 'https']))
    host = draw(st.from_regex(r'[a-z]{3,10}\.(com|org)', fullmatch=True))
    path = draw(st.one_of(
        st.none(),
        st.just(''),
        st.from_regex(r'/[a-z0-9/_-]{0,30}', fullmatch=True)
    ))
    return {'scheme': scheme, 'host': host, 'path': path}

@given(url_build_components())
@settings(max_examples=500)
def test_anyurl_build_path_roundtrip(components):
    url = AnyUrl.build(**components)

    if components['path'] is not None and components['path'] != '':
        assert url.path == components['path'], \
            f"path mismatch: expected {components['path']!r}, got {url.path!r}"
```

**Failing input**: `{'scheme': 'http', 'host': 'a.aa', 'path': '/', 'port': 1}`

## Reproducing the Bug

```python
from pydantic.networks import AnyUrl

url = AnyUrl.build(scheme='http', host='example.com', port=8080, path='/')
print(f"Input path: '/'")
print(f"Output path: {url.path!r}")
print(f"Full URL: {url}")

components = {'scheme': 'http', 'host': 'example.com', 'path': '/'}
url1 = AnyUrl.build(**components)

extracted_path = url1.path
url2 = AnyUrl.build(scheme='http', host='example.com', path=extracted_path)

print(f"\nOriginal path: '/'")
print(f"After one round-trip: {url1.path!r}")
print(f"After two round-trips: {url2.path!r}")
```

**Output**:
```
Input path: '/'
Output path: '//'
Full URL: http://example.com:8080//

Original path: '/'
After one round-trip: '//'
After two round-trips: '///'
```

## Why This Is A Bug

When `path='/'` is passed to `AnyUrl.build()`, the resulting URL's `.path` property returns `'//'` instead of `'/'`. This is incorrect behavior that violates the round-trip property.

The bug compounds with repeated round-trips:
- Input: `path='/'`
- After 1st build: `path='//'`
- After 2nd build: `path='///'`
- After 3rd build: `path='////'`

This clearly indicates a path duplication bug in the `build()` method.

The `.path` property is documented as "The path part of the URL, or `None`", with the example `/path` in `https://user:pass@host:port/path?query#fragment`. A path of `/` should remain `/`, not become `//`.

Additionally, `//` in URLs has special meaning (protocol-relative URLs), so this bug could cause significant issues:
```python
url = AnyUrl.build(scheme='http', host='example.com', path='/')
print(url)  # http://example.com// - the // is confusing and incorrect
```

## Fix

The root cause appears to be in how `build()` concatenates the URL components. When the path is `/`, it's likely being appended to a host that already ends with `/`, or vice versa.

Hypothetical fix (actual implementation is in pydantic-core):
```diff
--- a/pydantic_core/url.py
+++ b/pydantic_core/url.py
@@ -XX,X +XX,X @@ class Url:
     def build(..., path: str | None = None, ...):
-        url_string = f"{scheme}://{host}{path or ''}"
+        # Prevent double-slash when path is '/'
+        if path == '/':
+            url_string = f"{scheme}://{host}/"
+        elif path:
+            url_string = f"{scheme}://{host}{path}"
+        else:
+            url_string = f"{scheme}://{host}"
```

The fix should ensure that `path='/'` results in exactly one slash in the URL string.