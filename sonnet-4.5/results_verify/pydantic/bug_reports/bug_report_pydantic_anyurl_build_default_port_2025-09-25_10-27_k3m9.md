# Bug Report: pydantic.networks.AnyUrl.build() Inserts Default Port

**Target**: `pydantic.networks.AnyUrl.build()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`AnyUrl.build()` violates the round-trip property by inserting default ports (80 for HTTP, 443 for HTTPS) when `port=None` is explicitly provided. This causes the extracted `.port` property to return the default port instead of `None`, breaking the invariant that `build(**components).port == components['port']`.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
from pydantic.networks import AnyUrl

@st.composite
def url_build_components(draw):
    scheme = draw(st.sampled_from(['http', 'https']))
    host = draw(st.from_regex(r'[a-z]{3,10}\.(com|org)', fullmatch=True))
    port = draw(st.one_of(st.none(), st.integers(min_value=1, max_value=65535)))
    return {'scheme': scheme, 'host': host, 'port': port}

@given(url_build_components())
@settings(max_examples=500)
def test_anyurl_build_port_roundtrip(components):
    url = AnyUrl.build(**components)
    assert url.port == components['port'], \
        f"port mismatch: expected {components['port']}, got {url.port}"
```

**Failing input**: `{'scheme': 'http', 'host': 'a.aa', 'port': None}`

## Reproducing the Bug

```python
from pydantic.networks import AnyUrl

url = AnyUrl.build(scheme='http', host='example.com', port=None)
print(f"Input port: None")
print(f"Output port: {url.port}")
print(f"URL: {url}")

url_https = AnyUrl.build(scheme='https', host='example.com', port=None)
print(f"\nHTTPS input port: None")
print(f"HTTPS output port: {url_https.port}")
```

**Output**:
```
Input port: None
Output port: 80
URL: http://example.com/

HTTPS input port: None
HTTPS output port: 443
```

## Why This Is A Bug

The `AnyUrl.build()` method's docstring states that `port` is "The port part of the URL, or omit for no port." The `.port` property is documented as returning `int | None`, with the description "The port part of the URL, or `None`."

When a user explicitly passes `port=None` to `build()`, they expect the resulting URL's `.port` property to be `None`. However, pydantic inserts the scheme's default port (80 for HTTP, 443 for HTTPS) instead.

This violates the fundamental round-trip property:
```python
url = AnyUrl.build(scheme='http', host='example.com', port=None)
extracted_port = url.port  # Should be None, but is 80

url2 = AnyUrl.build(scheme='http', host='example.com', port=extracted_port)
# url2.port is still 80, but it should be None to match the original input
```

The user cannot distinguish between:
1. A URL built with `port=None` (no port specified)
2. A URL built with `port=80` for HTTP (explicit default port)

Both result in `.port == 80`, making it impossible to reconstruct the original input.

## Fix

The fix depends on the intended semantics. If `port=None` should mean "use default port", then the `.port` property should continue returning the default port, but this should be clearly documented. However, this breaks the round-trip property.

Alternatively, if `port=None` should mean "no port specified", then `build()` should not insert the default port into the underlying URL, and `.port` should return `None`:

```diff
--- a/pydantic_core/url.py (hypothetical - actual fix would be in pydantic-core)
+++ b/pydantic_core/url.py
@@ -XX,X +XX,X @@ class Url:
     def build(..., port: int | None = None, ...):
-        # Current behavior: insert default port if None
-        if port is None:
-            port = default_port_for_scheme(scheme)
+        # Proposed behavior: keep port as None if not specified
+        # Only use default port for display/serialization, not storage
```

This would preserve the round-trip property while still allowing default ports in string representations.