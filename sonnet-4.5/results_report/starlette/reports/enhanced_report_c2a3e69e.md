# Bug Report: Starlette URL.replace() IndexError on Empty Netloc

**Target**: `starlette.datastructures.URL.replace`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `URL.replace()` method crashes with an `IndexError` when called on a URL with an empty netloc (e.g., `"http:///path"`) and attempting to replace hostname-related parameters like `port`, `username`, or `password`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from starlette.datastructures import URL

@given(
    st.sampled_from(["http", "https", "ws", "wss"]),
    st.text(min_size=1),
    st.integers(min_value=1, max_value=65535) | st.none()
)
def test_url_replace_with_empty_netloc(scheme, path, port):
    url_str = f"{scheme}:///{path}"
    url = URL(url_str)

    if port is not None:
        new_url = url.replace(port=port)
        assert new_url.port == port
```

<details>

<summary>
**Failing input**: `test_url_replace_with_empty_netloc(scheme='http', path='0', port=1)`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 18, in <module>
    test_url_replace_with_empty_netloc()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 5, in test_url_replace_with_empty_netloc
    st.sampled_from(["http", "https", "ws", "wss"]),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 14, in test_url_replace_with_empty_netloc
    new_url = url.replace(port=port)
  File "/home/npc/miniconda/lib/python3.13/site-packages/starlette/datastructures.py", line 121, in replace
    if hostname[-1] != "]":
       ~~~~~~~~^^^^
IndexError: string index out of range
Falsifying example: test_url_replace_with_empty_netloc(
    scheme='http',
    path='0',
    port=1,
)
```
</details>

## Reproducing the Bug

```python
from starlette.datastructures import URL

# Test case that causes IndexError
url = URL("http:///path")
print(f"Created URL: {url}")
print(f"URL components: scheme='{url.scheme}', netloc='{url.netloc}', path='{url.path}'")
print(f"Attempting to replace port...")

try:
    new_url = url.replace(port=8080)
    print(f"Success: {new_url}")
except IndexError as e:
    print(f"IndexError: {e}")
    import traceback
    traceback.print_exc()
```

<details>

<summary>
IndexError: string index out of range
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/52/repo.py", line 10, in <module>
    new_url = url.replace(port=8080)
  File "/home/npc/miniconda/lib/python3.13/site-packages/starlette/datastructures.py", line 121, in replace
    if hostname[-1] != "]":
       ~~~~~~~~^^^^
IndexError: string index out of range
Created URL: http:///path
URL components: scheme='http', netloc='', path='/path'
Attempting to replace port...
IndexError: string index out of range

```
</details>

## Why This Is A Bug

This bug violates expected behavior because URLs with empty netloc are valid according to RFC 3986, yet the `URL.replace()` method crashes when attempting to modify them. The bug occurs in the URL.replace() method at lines 125-130 of `/home/npc/miniconda/lib/python3.13/site-packages/starlette/datastructures.py`:

1. When a URL like `"http:///path"` is parsed, `self.netloc` is an empty string `""`
2. The code executes `hostname = netloc.rpartition("@")[2]`, which returns `""` for empty netloc
3. The code then attempts `if hostname[-1] != "]"` to check for IPv6 bracket notation
4. Accessing `hostname[-1]` on an empty string raises `IndexError: string index out of range`

The code assumes that `hostname` will always have at least one character, but this assumption fails when the netloc is empty. URLs with empty netloc are not only valid per RFC 3986 but are commonly used in practice (e.g., `file:///` URLs for local files). Python's standard `urllib` library handles these URLs correctly without crashing.

## Relevant Context

- **RFC 3986 Compliance**: URLs with empty netloc (authority component) are syntactically valid. The triple slash format (`scheme:///path`) indicates an empty authority followed by an absolute path.
- **Common Use Cases**: File URLs commonly use this format (`file:///home/user/file.txt`) where the netloc is intentionally empty to represent local filesystem access.
- **Standard Library Behavior**: Python's `urllib.parse` correctly handles these URLs:
  ```python
  >>> from urllib.parse import urlsplit
  >>> urlsplit("http:///path")
  SplitResult(scheme='http', netloc='', path='/path', query='', fragment='')
  ```
- **Source Code Location**: Bug is at `/home/npc/miniconda/lib/python3.13/site-packages/starlette/datastructures.py:121`

## Proposed Fix

```diff
     def replace(self, **kwargs: Any) -> URL:
         if "username" in kwargs or "password" in kwargs or "hostname" in kwargs or "port" in kwargs:
             hostname = kwargs.pop("hostname", None)
             port = kwargs.pop("port", self.port)
             username = kwargs.pop("username", self.username)
             password = kwargs.pop("password", self.password)

             if hostname is None:
                 netloc = self.netloc
                 _, _, hostname = netloc.rpartition("@")

-                if hostname[-1] != "]":
+                if hostname and hostname[-1] != "]":
                     hostname = hostname.rsplit(":", 1)[0]

             netloc = hostname
             if port is not None:
                 netloc += f":{port}"
             if username is not None:
                 userpass = username
                 if password is not None:
                     userpass += f":{password}"
                 netloc = f"{userpass}@{netloc}"

             kwargs["netloc"] = netloc

         components = self.components._replace(**kwargs)
         return self.__class__(components.geturl())
```