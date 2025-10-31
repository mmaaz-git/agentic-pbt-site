# Bug Report: pandas.io.common.is_url - Crashes on Malformed URLs

**Target**: `pandas.io.common.is_url`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `is_url` function crashes with a `ValueError` when given certain malformed URLs with unmatched brackets (e.g., `http://[`), instead of returning `False` as expected for invalid URLs.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.io.common import is_url


@given(st.text())
def test_is_url_handles_arbitrary_input(url_path):
    full_url = f"http://{url_path}"
    result = is_url(full_url)
    assert isinstance(result, bool)
```

**Failing input**: `is_url("http://[")`

## Reproducing the Bug

```python
from pandas.io.common import is_url

is_url("http://[")
```

Expected: Function should return `False` for malformed URLs.

Actual: Raises `ValueError: Invalid IPv6 URL`

## Why This Is A Bug

The function's docstring states that it should "Check to see if a URL has a valid protocol" and return a boolean value. The function is designed to validate URLs, so it should handle malformed URLs gracefully by returning `False` rather than raising an exception.

The crash occurs because `is_url` calls `parse_url` (from `urllib.parse`) which raises a `ValueError` for certain malformed URLs, and this exception is not caught.

Users who pass untrusted or user-provided URLs to this function would experience unexpected crashes instead of getting a `False` return value for invalid URLs.

## Fix

The fix wraps the `parse_url` call in a try-except block to catch `ValueError` exceptions from malformed URLs and return `False` instead of propagating the exception.
