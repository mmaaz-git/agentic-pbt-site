# Bug Report: anyio.abc._sockets._validate_socket Misleading Error Messages

**Target**: `anyio.abc._sockets._validate_socket`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `_validate_socket` function produces misleading error messages when given an invalid file descriptor with `require_connected=True` or `require_bound=True`. It incorrectly reports connection/binding issues when the real problem is that the file descriptor is invalid.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st, assume
import errno
import socket
from anyio.abc._sockets import _validate_socket


@given(st.integers(min_value=100, max_value=999999))
@settings(max_examples=200)
def test_validate_socket_error_messages_accurate(fd):
    try:
        test_sock = socket.socket(fileno=fd)
        test_sock.close()
    except OSError as e:
        assume(e.errno != errno.ENOTSOCK)
        actual_error = e
    else:
        assume(False)

    try:
        _validate_socket(fd, socket.SOCK_STREAM, require_connected=True)
        assert False, "Should have raised an exception"
    except ValueError as e:
        assert "must be connected" not in str(e), (
            f"Misleading error: got '{e}' but real issue is "
            f"OSError(errno={actual_error.errno})"
        )
    except OSError:
        pass
```

**Failing input**: Any invalid file descriptor (e.g., `99999`, `12345`)

## Reproducing the Bug

```python
import socket
from anyio.abc._sockets import _validate_socket

invalid_fd = 99999

try:
    _validate_socket(invalid_fd, socket.SOCK_STREAM, require_connected=True)
except ValueError as e:
    print(e)
```

**Output**: `ValueError: the socket must be connected`

**Expected**: Should either raise `OSError` with the original error (e.g., "Bad file descriptor") or a `ValueError` that accurately describes the problem.

## Why This Is A Bug

The error message "the socket must be connected" is misleading when the file descriptor doesn't even refer to a valid socket. This violates the API contract by providing inaccurate diagnostic information to users.

When `socket.socket(fileno=fd)` fails with `OSError(errno=EBADF)` (bad file descriptor), the code catches this exception and, if `require_connected=True`, converts it to `ValueError("the socket must be connected")`. This suggests to users that they have a valid socket that isn't connected, when in fact they don't have a valid socket at all.

The same issue occurs with `require_bound=True`, which produces "the socket must be bound to a local address" for invalid file descriptors.

## Fix

```diff
--- a/anyio/abc/_sockets.py
+++ b/anyio/abc/_sockets.py
@@ -44,14 +44,8 @@ def _validate_socket(
     if isinstance(sock_or_fd, int):
         try:
             sock = socket.socket(fileno=sock_or_fd)
         except OSError as exc:
-            if exc.errno == errno.ENOTSOCK:
-                raise ValueError(
-                    "the file descriptor does not refer to a socket"
-                ) from exc
-            elif require_connected:
-                raise ValueError("the socket must be connected") from exc
-            elif require_bound:
-                raise ValueError("the socket must be bound to a local address") from exc
-            else:
-                raise
+            raise ValueError(
+                "the file descriptor does not refer to a socket"
+            ) from exc
     elif isinstance(sock_or_fd, socket.socket):
         sock = sock_or_fd
```

This fix simplifies the error handling by treating all `OSError` exceptions from `socket.socket(fileno=...)` as indicating an invalid file descriptor, which is more accurate. The connection and binding checks should only happen after successfully creating the socket object (which they already do at lines 66-82).