# Bug Report: starlette URL.replace IndexError with empty hostname

**Target**: `starlette.datastructures.URL.replace`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `URL.replace()` method in Starlette crashes with an `IndexError` when called on URLs containing empty hostnames (e.g., `http://@/path`), which are technically valid URL formats according to RFC 3986.

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

# Run the test
test_url_replace_port_with_empty_hostname()
```

<details>

<summary>
**Failing input**: `port=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/41/hypo.py", line 13, in <module>
    test_url_replace_port_with_empty_hostname()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/41/hypo.py", line 6, in test_url_replace_port_with_empty_hostname
    @settings(max_examples=100)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/41/hypo.py", line 9, in test_url_replace_port_with_empty_hostname
    result = url.replace(port=port)
  File "/home/npc/miniconda/lib/python3.13/site-packages/starlette/datastructures.py", line 121, in replace
    if hostname[-1] != "]":
       ~~~~~~~~^^^^
IndexError: string index out of range
Falsifying example: test_url_replace_port_with_empty_hostname(
    port=1,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
from starlette.datastructures import URL

# Test case that causes the crash
url = URL("http://@/path")
result = url.replace(port=8000)
print(f"Result: {result}")
```

<details>

<summary>
IndexError: string index out of range
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/41/repo.py", line 5, in <module>
    result = url.replace(port=8000)
  File "/home/npc/miniconda/lib/python3.13/site-packages/starlette/datastructures.py", line 121, in replace
    if hostname[-1] != "]":
       ~~~~~~~~^^^^
IndexError: string index out of range
```
</details>

## Why This Is A Bug

This violates expected behavior because the `URL.replace()` method should handle all valid URL formats gracefully, even edge cases. The URL `http://@/path` is syntactically valid according to RFC 3986, which permits empty host components after the userinfo separator `@`.

The crash occurs in `datastructures.py` at line 121 when the code attempts to check if the hostname ends with `]` (for IPv6 addresses) without first verifying that the hostname is non-empty. When parsing `http://@/path`, the netloc is `"@"`, and after `netloc.rpartition("@")`, the hostname becomes an empty string `""`. Accessing `hostname[-1]` on an empty string raises an `IndexError`.

The library should either successfully process the URL replacement or raise a meaningful validation error, not crash with an unhandled `IndexError` that reveals internal implementation details.

## Relevant Context

The bug manifests when calling `URL.replace()` with any of these parameters: `port`, `username`, `password`, or `hostname` on URLs with empty hostnames. The issue is in the hostname extraction logic within the `replace()` method.

According to RFC 3986 Section 3.2.2, the authority component has this structure:
```
authority = [ userinfo "@" ] host [ ":" port ]
```
Where the host can be a registered name that "may be empty (zero length)". This means URLs like `http://@/path` are technically valid, though admittedly unusual.

Python's standard `urllib.parse` library handles such URLs without crashing:
```python
from urllib.parse import urlparse, urlunparse
parsed = urlparse("http://@/path")
# Works without error
```

Link to affected code: [starlette/datastructures.py line 121](https://github.com/encode/starlette/blob/master/starlette/datastructures.py#L121)

## Proposed Fix

```diff
--- a/starlette/datastructures.py
+++ b/starlette/datastructures.py
@@ -118,7 +118,7 @@ class URL:
             netloc = self.netloc
             _, _, hostname = netloc.rpartition("@")

-            if hostname[-1] != "]":
+            if hostname and hostname[-1] != "]":
                 hostname = hostname.rsplit(":", 1)[0]

         netloc = hostname
```