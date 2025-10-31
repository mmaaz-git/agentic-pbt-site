# Bug Report: pandas.io.common.is_url Crashes on Malformed IPv6 URLs

**Target**: `pandas.io.common.is_url`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `is_url` function crashes with `ValueError: Invalid IPv6 URL` when given malformed URLs containing unmatched brackets, instead of returning `False` as documented.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas.io.common as common


@given(scheme=st.sampled_from(['http', 'https', 'ftp', 'file']),
       domain=st.text(min_size=1))
def test_is_url_returns_bool(scheme, domain):
    url = f"{scheme}://{domain}"
    result = common.is_url(url)
    assert isinstance(result, bool)
```

**Failing input**: `scheme='http', domain='['`

## Reproducing the Bug

```python
import pandas.io.common as common

url = "http://["
result = common.is_url(url)
```

**Output:**
```
ValueError: Invalid IPv6 URL
```

## Why This Is A Bug

The `is_url` function is documented to "Check to see if a URL has a valid protocol" and return a boolean: "If `url` has a valid protocol return True otherwise False." However, it crashes on malformed URLs instead of returning `False`. This violates the function's contract of always returning a boolean value.

A validation function should handle invalid inputs gracefully by returning `False`, not by raising an exception. Users may pass arbitrary strings to check if they are URLs, and those strings may be malformed.

## Fix

```diff
--- a/pandas/io/common.py
+++ b/pandas/io/common.py
@@ -172,7 +172,10 @@ def is_url(url: object) -> bool:
     """
     if not isinstance(url, str):
         return False
-    return parse_url(url).scheme in _VALID_URLS
+    try:
+        return parse_url(url).scheme in _VALID_URLS
+    except ValueError:
+        return False
```