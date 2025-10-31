# Bug Report: starlette.datastructures.URL.replace IndexError on Empty Netloc

**Target**: `starlette.datastructures.URL.replace`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `URL.replace()` method crashes with `IndexError: string index out of range` when attempting to replace URL components (port, username, etc.) on URLs with empty or minimal netloc values such as `http://@/path`, `http:///path`, or `http://user@/path`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from starlette.datastructures import URL

@given(st.integers(min_value=1, max_value=65535))
@settings(max_examples=100)
def test_url_replace_port_should_not_crash(port):
    test_urls = [
        "http://@/path",
        "http:///path",
        "http://user@/path",
        "http://user:pass@/path",
    ]

    for url_str in test_urls:
        url = URL(url_str)
        new_url = url.replace(port=port)
        assert isinstance(new_url, URL)
```

**Failing input**: Any of the test URLs with any port number, e.g., `URL("http://@/path").replace(port=8080)`

## Reproducing the Bug

```python
from starlette.datastructures import URL

url = URL("http://@/path")
new_url = url.replace(port=8080)
```

This crashes with:
```
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "starlette/datastructures.py", line 129, in replace
    if hostname[-1] != "]":
       ^^^^^^^^^^^^
IndexError: string index out of range
```

## Why This Is A Bug

The bug occurs in `starlette/datastructures.py` at line 125-130:

```python
if hostname is None:
    netloc = self.netloc
    _, _, hostname = netloc.rpartition("@")

if hostname[-1] != "]":  # Line 129 - CRASHES HERE
    hostname = hostname.rsplit(":", 1)[0]
```

When the netloc is `"@"`, `""`, or `"user@"`, the `rpartition("@")` returns an empty string for `hostname`. Accessing `hostname[-1]` on an empty string raises `IndexError`.

This affects valid (though unusual) URLs:
- `http://@/path` - URL with empty userinfo and empty host
- `http:///path` - URL with empty netloc
- `http://user@/path` - URL with username but no host

These URLs may be generated programmatically or appear in edge cases. The method should handle them gracefully rather than crashing.

## Fix

Add a check for empty hostname before accessing its last character:

```diff
--- a/starlette/datastructures.py
+++ b/starlette/datastructures.py
@@ -126,7 +126,7 @@ class URL:
                 netloc = self.netloc
                 _, _, hostname = netloc.rpartition("@")

-            if hostname[-1] != "]":
+            if hostname and hostname[-1] != "]":
                 hostname = hostname.rsplit(":", 1)[0]

             netloc = hostname
```

This ensures that:
1. Empty hostnames are handled without crashing
2. IPv6 addresses in brackets (ending with `]`) are still handled correctly
3. Regular hostnames with ports are still split correctly