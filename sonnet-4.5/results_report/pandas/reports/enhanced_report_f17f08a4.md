# Bug Report: pandas.io.common.is_url Raises ValueError on Malformed URLs

**Target**: `pandas.io.common.is_url`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `is_url` function violates its documented contract by raising a `ValueError` when given URLs with malformed IPv6 addresses instead of returning `False` as promised in its docstring.

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

# Run the test to find the failing case
if __name__ == "__main__":
    import traceback
    try:
        test_is_url_accepts_valid_schemes()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed with assertion error: {e}")
        print(traceback.format_exc())
    except Exception as e:
        print(f"Test failed with exception: {type(e).__name__}: {e}")
        print(traceback.format_exc())
```

<details>

<summary>
**Failing input**: `scheme='git', path='['`
</summary>
```
Test failed with exception: ValueError: Invalid IPv6 URL
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 19, in <module>
    test_is_url_accepts_valid_schemes()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 10, in test_is_url_accepts_valid_schemes
    def test_is_url_accepts_valid_schemes(scheme, path):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 12, in test_is_url_accepts_valid_schemes
    result = pandas.io.common.is_url(url)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/common.py", line 175, in is_url
    return parse_url(url).scheme in _VALID_URLS
           ~~~~~~~~~^^^^^
  File "/home/npc/miniconda/lib/python3.13/urllib/parse.py", line 395, in urlparse
    splitresult = urlsplit(url, scheme, allow_fragments)
  File "/home/npc/miniconda/lib/python3.13/urllib/parse.py", line 514, in urlsplit
    raise ValueError("Invalid IPv6 URL")
ValueError: Invalid IPv6 URL
Falsifying example: test_is_url_accepts_valid_schemes(
    scheme='git',  # or any other generated value
    path='[',
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/urllib/parse.py:514
```
</details>

## Reproducing the Bug

```python
import pandas.io.common

# This is the minimal failing test case from the bug report
# The URL contains a malformed IPv6 address bracket '['
url = "nfs://["
try:
    result = pandas.io.common.is_url(url)
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception raised: {type(e).__name__}: {e}")
```

<details>

<summary>
ValueError raised when checking malformed URL
</summary>
```
Exception raised: ValueError: Invalid IPv6 URL
```
</details>

## Why This Is A Bug

This is a contract violation bug. The function's docstring explicitly states:

```
Returns
-------
isurl : bool
    If `url` has a valid protocol return True otherwise False.
```

The docstring establishes a clear contract: the function will return a boolean value - `True` if the URL has a valid protocol, `False` otherwise. There is no mention of exceptions being raised for malformed URLs.

When given a URL like `"nfs://["` (or any URL with a single opening bracket `[` that looks like a malformed IPv6 address), the function raises a `ValueError` instead of returning `False`. This violates the documented API contract.

The issue occurs because:
1. The `is_url` function calls `parse_url(url)` (which is `urllib.parse.urlparse`)
2. `urlparse` internally calls `urlsplit`
3. `urlsplit` raises a `ValueError` when it encounters what appears to be a malformed IPv6 address (a `[` without a matching `]`)
4. The `is_url` function does not catch this exception, allowing it to propagate to the caller

## Relevant Context

- The bug affects any URL scheme combined with a malformed IPv6 address pattern (single `[` bracket)
- The `_VALID_URLS` set contains 31 valid URL schemes from urllib.parse
- Source code location: `/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/common.py:175`
- The function is likely used throughout pandas for URL validation in I/O operations
- Documentation: https://pandas.pydata.org/docs/reference/api/pandas.io.common.is_url.html

## Proposed Fix

Wrap the `parse_url` call in a try-except block to catch `ValueError` and return `False` as per the documented contract:

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
+        # Handle malformed URLs (e.g., invalid IPv6 addresses)
+        return False
```