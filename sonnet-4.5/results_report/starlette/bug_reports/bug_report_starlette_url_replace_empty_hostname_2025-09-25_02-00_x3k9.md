# Bug Report: starlette URL.replace IndexError with empty hostname

**Target**: `starlette.datastructures.URL.replace`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `URL.replace()` method crashes with an `IndexError` when called on URLs with empty hostnames (e.g., `http://@/path`). This occurs when trying to replace URL components like port, username, or password.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from starlette.datastructures import URL


@given(port=st.integers(min_value=1, max_value=65535))
@settings(max_examples=100)
def test_url_replace_port_with_empty_hostname(port):
    url = URL("http://@/path")
    result = url.replace(port=port)
    assert isinstance(result, URL)
```

**Failing input**: `port=1` (or any valid port number)

## Reproducing the Bug

```python
from starlette.datastructures import URL

url = URL("http://@/path")
result = url.replace(port=8000)
```

**Output**:
```
IndexError: string index out of range
```

**Traceback**:
```
File "starlette/datastructures.py", line 129, in replace
    if hostname[-1] != "]":
       ^^^^^^^^^^^^
IndexError: string index out of range
```

## Why This Is A Bug

The URL `http://@/path` is a valid URL structure where the username/password separator `@` is present but the hostname is empty. The `replace()` method should handle this gracefully instead of crashing.

The bug occurs in `datastructures.py` at lines 125-130:

```python
if hostname is None:
    netloc = self.netloc
    _, _, hostname = netloc.rpartition("@")

    if hostname[-1] != "]":  # BUG: hostname can be empty!
        hostname = hostname.rsplit(":", 1)[0]
```

When `netloc = "@"`, the `rpartition("@")` returns `('', '@', '')`, so `hostname = ""`. Attempting to access `hostname[-1]` on an empty string raises an `IndexError`.

## Fix

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

This fix checks that `hostname` is not empty before attempting to access its last character.