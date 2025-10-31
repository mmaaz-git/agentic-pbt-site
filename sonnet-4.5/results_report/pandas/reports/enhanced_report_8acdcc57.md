# Bug Report: pandas.io.common.is_url Crashes on Malformed IPv6 URLs Instead of Returning False

**Target**: `pandas.io.common.is_url`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `is_url` function crashes with `ValueError: Invalid IPv6 URL` when given malformed URLs containing unmatched brackets, violating its documented contract of always returning a boolean value.

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

if __name__ == "__main__":
    test_is_url_returns_bool()
```

<details>

<summary>
**Failing input**: `scheme='http', domain='['`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/8/hypo.py", line 13, in <module>
    test_is_url_returns_bool()
    ~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/8/hypo.py", line 6, in test_is_url_returns_bool
    domain=st.text(min_size=1))
            ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/8/hypo.py", line 9, in test_is_url_returns_bool
    result = common.is_url(url)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/common.py", line 175, in is_url
    return parse_url(url).scheme in _VALID_URLS
           ~~~~~~~~~^^^^^
  File "/home/npc/miniconda/lib/python3.13/urllib/parse.py", line 395, in urlparse
    splitresult = urlsplit(url, scheme, allow_fragments)
  File "/home/npc/miniconda/lib/python3.13/urllib/parse.py", line 514, in urlsplit
    raise ValueError("Invalid IPv6 URL")
ValueError: Invalid IPv6 URL
Falsifying example: test_is_url_returns_bool(
    scheme='http',  # or any other generated value
    domain='[',
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/urllib/parse.py:514
```
</details>

## Reproducing the Bug

```python
import pandas.io.common as common

url = "http://["
result = common.is_url(url)
print(f"Result: {result}")
```

<details>

<summary>
ValueError: Invalid IPv6 URL
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/8/repo.py", line 4, in <module>
    result = common.is_url(url)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/common.py", line 175, in is_url
    return parse_url(url).scheme in _VALID_URLS
           ~~~~~~~~~^^^^^
  File "/home/npc/miniconda/lib/python3.13/urllib/parse.py", line 395, in urlparse
    splitresult = urlsplit(url, scheme, allow_fragments)
  File "/home/npc/miniconda/lib/python3.13/urllib/parse.py", line 514, in urlsplit
    raise ValueError("Invalid IPv6 URL")
ValueError: Invalid IPv6 URL
```
</details>

## Why This Is A Bug

This violates the expected behavior of `pandas.io.common.is_url` in several critical ways:

1. **Contract Violation**: The function's docstring explicitly states: "Returns: isurl : bool - If `url` has a valid protocol return True otherwise False." The documentation promises a boolean return value for all inputs, but the function raises an exception instead.

2. **Semantic Expectations**: The function follows Python's `is_*` naming convention (like `isinstance`, `isdigit`, `isalpha`), which establishes a clear expectation that validation functions should return `False` for invalid inputs rather than raising exceptions. This is a fundamental pattern in Python's standard library.

3. **Implementation Inconsistency**: The function already handles non-string inputs gracefully by returning `False` (lines 173-174), showing that defensive programming was intended. However, it fails to handle the case where `parse_url` might raise an exception.

4. **Real-World Impact**: Users may pass arbitrary strings from user input, configuration files, or external data sources to check if they are valid URLs. These strings could contain malformed patterns like unmatched brackets. The current behavior forces users to wrap every call in a try-except block, which defeats the purpose of having a validation function.

5. **Documentation vs Implementation Mismatch**: The underlying `urllib.parse.urlparse` documentation explicitly warns that "Unmatched square brackets in the netloc attribute will raise a ValueError," but `pandas.io.common.is_url` does not mention this exception in its documentation, creating a hidden failure mode.

## Relevant Context

The bug occurs at line 175 in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/common.py`:

```python
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
    return parse_url(url).scheme in _VALID_URLS  # Line 175 - raises ValueError
```

The Python standard library's `urllib.parse` module documentation ([link](https://docs.python.org/3/library/urllib.parse.html)) states:
- "Following the syntax specifications in RFC 1808, urlparse recognizes a netloc only if it is properly introduced by '//'."
- "Unmatched square brackets in the netloc attribute will raise a ValueError."

This is a known behavior of `urlparse`, but `pandas.io.common.is_url` should handle this exception internally since it promises to return a boolean for all inputs.

Other malformed URLs that trigger this bug:
- `"http://]"` - unmatched closing bracket
- `"https://[:"` - incomplete IPv6 address
- `"ftp://[::"` - incomplete IPv6 address
- Any URL with unmatched square brackets in the netloc portion

## Proposed Fix

```diff
--- a/pandas/io/common.py
+++ b/pandas/io/common.py
@@ -172,5 +172,8 @@ def is_url(url: object) -> bool:
     """
     if not isinstance(url, str):
         return False
-    return parse_url(url).scheme in _VALID_URLS
+    try:
+        return parse_url(url).scheme in _VALID_URLS
+    except ValueError:
+        return False
```