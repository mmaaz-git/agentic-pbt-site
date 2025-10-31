# Bug Report: pandas.io.common.is_url Raises ValueError on Malformed URLs

**Target**: `pandas.io.common.is_url`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `is_url` function violates its documented contract by raising `ValueError` on malformed URLs instead of returning `False`. The docstring explicitly states it should return a boolean value, but certain inputs cause it to raise an exception.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas.io.common

VALID_SCHEMES = {'wss', 'sips', 'svn', 'git', 'tel', 'rtspu', 'rtsps', 'nfs', 'shttp', 'ws',
                 'imap', 'mms', 'prospero', 'sip', 'rsync', 'wais', 'ftp', 'itms-services',
                 'gopher', 'hdl', 'svn+ssh', 'https', 'sftp', 'snews', 'telnet', 'file',
                 'git+ssh', 'rtsp', 'nntp', 'http'}

@given(st.sampled_from(list(VALID_SCHEMES)), st.text())
def test_is_url_accepts_valid_schemes(scheme, path):
    url = f"{scheme}://{path}"
    result = pandas.io.common.is_url(url)
    assert result is True
```

**Failing input**: `scheme='nfs', path='['` (produces URL `'nfs://['`)

## Reproducing the Bug

```python
import pandas.io.common

url = "nfs://["
result = pandas.io.common.is_url(url)
```

Expected: `False`
Actual: `ValueError: Invalid IPv6 URL`

## Why This Is A Bug

The function's docstring clearly states:

```
Returns
-------
isurl : bool
    If `url` has a valid protocol return True otherwise False.
```

This establishes a contract that the function will return a boolean value. However, when given a malformed URL like `"nfs://["`, the function raises a `ValueError` instead of returning `False`. This violates the documented API contract.

The bug occurs because `is_url` calls `parse_url` (from `urllib.parse.urlparse`) without handling exceptions. When `urlparse` encounters a malformed IPv6 address in the URL, it raises `ValueError`, which propagates up uncaught.

## Fix

Wrap the `parse_url` call in a try-except block to catch `ValueError` and return `False`:

```diff
def is_url(url: object) -> bool:
    """
    Check to see if a URL has a valid protocol.

    Parameters
    ----------
    url : str or unicode

    Returns
    -------
    isurl : bool
        If `url` has a valid protocol return True otherwise False.
    """
    if not isinstance(url, str):
        return False
-    return parse_url(url).scheme in _VALID_URLS
+    try:
+        return parse_url(url).scheme in _VALID_URLS
+    except ValueError:
+        return False
```