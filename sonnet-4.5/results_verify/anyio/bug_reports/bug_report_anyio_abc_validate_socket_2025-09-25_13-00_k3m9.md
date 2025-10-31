# Bug Report: anyio.abc._validate_socket Misleading Error Messages for Invalid File Descriptors

**Target**: `anyio.abc._sockets._validate_socket`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_validate_socket` function produces misleading error messages when given an invalid or closed file descriptor with `require_connected=True` or `require_bound=True`. It incorrectly reports that "the socket must be connected" or "must be bound" when the actual issue is that the file descriptor is invalid/closed.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import socket
import pytest
from anyio.abc._sockets import _validate_socket


@given(sock_type=st.sampled_from([socket.SOCK_STREAM, socket.SOCK_DGRAM]))
def test_invalid_fd_with_require_connected_gives_accurate_error(sock_type):
    sock = socket.socket(socket.AF_INET, sock_type)
    fd = sock.fileno()
    sock.close()

    with pytest.raises(ValueError) as exc_info:
        _validate_socket(fd, sock_type, require_connected=True)

    assert "must be connected" not in str(exc_info.value), \
        "Error message incorrectly suggests connection issue when fd is invalid"
```

**Failing input**: Any closed/invalid file descriptor with `require_connected=True` or `require_bound=True`

## Reproducing the Bug

```python
import socket
from anyio.abc._sockets import _validate_socket

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
fd = sock.fileno()
sock.close()

try:
    _validate_socket(fd, socket.SOCK_STREAM, require_connected=True)
except ValueError as e:
    print(f"Error: {e}")

```

**Expected output**: Error message indicating the file descriptor is invalid/closed
**Actual output**: `ValueError: the socket must be connected`

The underlying error is `OSError: [Errno 9] Bad file descriptor`, but the code misinterprets this as a connection requirement failure.

## Why This Is A Bug

Lines 52-55 in `_sockets.py` catch `OSError` exceptions from socket construction (line 46) and assume the error is related to connection/binding requirements. This is incorrect because:

1. Socket construction can fail for many reasons (closed fd, invalid fd, insufficient permissions, etc.)
2. You cannot determine if a socket is connected or bound if you couldn't construct the socket object
3. The error message misleads users into thinking the socket exists but lacks a connection/binding, when actually the socket couldn't be created at all

The logic conflates two distinct failure modes:
- Failure to create a socket object from the file descriptor
- Failure of a valid socket to meet connection/binding requirements

## Fix

```diff
--- a/anyio/abc/_sockets.py
+++ b/anyio/abc/_sockets.py
@@ -48,10 +48,8 @@ def _validate_socket(
             if exc.errno == errno.ENOTSOCK:
                 raise ValueError(
                     "the file descriptor does not refer to a socket"
                 ) from exc
-            elif require_connected:
-                raise ValueError("the socket must be connected") from exc
-            elif require_bound:
-                raise ValueError("the socket must be bound to a local address") from exc
             else:
                 raise
     elif isinstance(sock_or_fd, socket.socket):
```

The fix removes lines 52-55 which incorrectly assume that non-ENOTSOCK errors during socket construction are due to connection/binding requirements. Instead, these errors should be re-raised as-is or wrapped with a more accurate message about the file descriptor being invalid.

Alternatively, if wrapping is desired:

```diff
--- a/anyio/abc/_sockets.py
+++ b/anyio/abc/_sockets.py
@@ -48,10 +48,8 @@ def _validate_socket(
             if exc.errno == errno.ENOTSOCK:
                 raise ValueError(
                     "the file descriptor does not refer to a socket"
                 ) from exc
-            elif require_connected:
-                raise ValueError("the socket must be connected") from exc
-            elif require_bound:
-                raise ValueError("the socket must be bound to a local address") from exc
             else:
-                raise
+                raise ValueError(
+                    "the file descriptor is invalid or closed"
+                ) from exc
     elif isinstance(sock_or_fd, socket.socket):
```