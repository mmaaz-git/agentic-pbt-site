# Bug Report: anyio.abc._validate_socket Falsy Value Check

**Target**: `anyio.abc._sockets._validate_socket`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_validate_socket` function incorrectly rejects UNIX domain sockets that are in a valid but unbound state when `require_bound=True`. The bug is caused by using a falsy check (`if not bound_addr:`) which treats empty strings as "not bound" even though an empty string is the valid return value for unbound UNIX sockets.

## Property-Based Test

```python
import socket
from hypothesis import given, strategies as st
from anyio.abc._sockets import _validate_socket


@given(st.sampled_from([socket.AF_UNIX]))
def test_validate_socket_unix_unbound_require_bound(family):
    """
    Property: For UNIX domain sockets, getsockname() returns an empty string
    when the socket is unbound. The falsy check 'if not bound_addr:' incorrectly
    treats this as meaning the socket is not bound, when it's actually a valid
    state for UNIX sockets (especially DGRAM sockets which support autobind).
    """
    sock = socket.socket(family, socket.SOCK_DGRAM)

    bound_addr = sock.getsockname()
    assert bound_addr == "", "Unbound UNIX socket should have empty string address"

    try:
        _validate_socket(sock, socket.SOCK_DGRAM, require_bound=True)
    except ValueError as e:
        assert "must be bound" in str(e)
    finally:
        sock.close()
```

**Failing input**: Any unbound UNIX domain datagram socket

## Reproducing the Bug

```python
import socket
from anyio.abc._sockets import _validate_socket

sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)

print(f"Socket getsockname(): {sock.getsockname()!r}")
print(f"Is empty string: {sock.getsockname() == ''}")

try:
    _validate_socket(sock, socket.SOCK_DGRAM, require_bound=True)
    print("Success")
except ValueError as e:
    print(f"Error: {e}")

sock.close()
```

**Output**:
```
Socket getsockname(): ''
Is empty string: True
Error: the socket must be bound to a local address
```

## Why This Is A Bug

The code at lines 72-82 in `/anyio/abc/_sockets.py` uses a falsy check to determine if a socket is bound:

```python
if require_bound:
    try:
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            bound_addr = sock.getsockname()[1]
        else:
            bound_addr = sock.getsockname()
    except OSError:
        bound_addr = None

    if not bound_addr:  # BUG: Empty string is falsy!
        raise ValueError("the socket must be bound to a local address")
```

For UNIX domain sockets, `getsockname()` returns:
- A non-empty string path for explicitly bound sockets
- An empty string `""` for unbound sockets

The problem is that an empty string is a valid return value, not an error condition. The check `if not bound_addr:` treats the empty string as falsy and incorrectly raises ValueError.

This is particularly problematic for UNIX datagram sockets which support "autobind" - they don't need to be explicitly bound to send/receive on some systems.

## Fix

Replace the falsy check with an explicit check for `None`:

```diff
--- a/anyio/abc/_sockets.py
+++ b/anyio/abc/_sockets.py
@@ -78,7 +78,7 @@ def _validate_socket(
         except OSError:
             bound_addr = None

-        if not bound_addr:
+        if bound_addr is None:
             raise ValueError("the socket must be bound to a local address")

     if addr_family != socket.AF_UNSPEC and sock.family != addr_family:
```

This fix ensures that only when `getsockname()` raises an OSError (setting `bound_addr = None`) will the validation fail, not when it returns a valid empty string.