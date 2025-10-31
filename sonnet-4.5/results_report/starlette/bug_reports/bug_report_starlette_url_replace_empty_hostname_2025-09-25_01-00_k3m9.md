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

**Failing input**: `URL("http:///path").replace(port=8080)`

## Reproducing the Bug

```python
from starlette.datastructures import URL

url = URL("http:///path")
new_url = url.replace(port=8080)
```

Output:
```
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "starlette/datastructures.py", line 129, in replace
    if hostname[-1] != "]":
IndexError: string index out of range
```

## Why This Is A Bug

The bug is in lines 125-130 of `starlette/datastructures.py`:

```python
if hostname is None:
    netloc = self.netloc
    _, _, hostname = netloc.rpartition("@")

    if hostname[-1] != "]":
        hostname = hostname.rsplit(":", 1)[0]
```

When a URL like `"http:///path"` is parsed:
1. `self.netloc` is an empty string `""`
2. `"".rpartition("@")` returns `("", "", "")`
3. `hostname` becomes `""`
4. `hostname[-1]` attempts to access the last character of an empty string
5. This raises `IndexError: string index out of range`

The code assumes that `hostname` will always have at least one character when checking for IPv6 bracket notation, but this assumption fails when the netloc is empty.

## Fix

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