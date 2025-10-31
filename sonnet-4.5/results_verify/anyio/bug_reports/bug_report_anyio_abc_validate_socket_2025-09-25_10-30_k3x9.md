# Bug Report: anyio.abc._validate_socket Invalid File Descriptor Handling

**Target**: `anyio.abc._sockets._validate_socket`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `_validate_socket` function inconsistently handles invalid file descriptors. It converts `ENOTSOCK` errors to `ValueError` but not `EBADF` errors, leading to inconsistent exception types for similar invalid inputs.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import socket
import pytest

@given(fd=st.integers(min_value=10000, max_value=999999))
def test_validate_socket_invalid_fd_raises_valueerror(fd):
    from anyio.abc._sockets import _validate_socket

    with pytest.raises(ValueError, match="does not refer to a socket"):
        _validate_socket(fd, socket.SOCK_STREAM)
```

**Failing input**: Any invalid file descriptor (e.g., `999999`)

## Reproducing the Bug

```python
import socket
import errno
import tempfile


def _validate_socket(sock_or_fd, sock_type, addr_family=socket.AF_UNSPEC, *, require_connected=False, require_bound=False):
    if isinstance(sock_or_fd, int):
        try:
            sock = socket.socket(fileno=sock_or_fd)
        except OSError as exc:
            if exc.errno == errno.ENOTSOCK:
                raise ValueError("the file descriptor does not refer to a socket") from exc
            elif require_connected:
                raise ValueError("the socket must be connected") from exc
            elif require_bound:
                raise ValueError("the socket must be bound to a local address") from exc
            else:
                raise
    else:
        sock = sock_or_fd

    sock.setblocking(False)
    return sock


with tempfile.NamedTemporaryFile() as f:
    try:
        _validate_socket(f.fileno(), socket.SOCK_STREAM)
    except ValueError:
        print("File FD: Raises ValueError ✓")

try:
    _validate_socket(999999, socket.SOCK_STREAM)
except ValueError:
    print("Invalid FD: Raises ValueError ✓")
except OSError as e:
    print(f"Invalid FD: Raises OSError ✗ (errno={e.errno})")
```

## Why This Is A Bug

The function's error handling is inconsistent:

1. **Valid FD for non-socket** (e.g., file): `OSError(errno=ENOTSOCK)` → Converted to `ValueError`
2. **Invalid FD** (e.g., 999999): `OSError(errno=EBADF)` → Re-raised as `OSError`

Both cases represent invalid input to a socket validation function, so they should raise the same exception type. The error message "the file descriptor does not refer to a socket" applies equally to both cases. This inconsistency violates the API contract and makes error handling unpredictable.

## Fix

```diff
--- a/anyio/abc/_sockets.py
+++ b/anyio/abc/_sockets.py
@@ -45,7 +45,7 @@ def _validate_socket(
         try:
             sock = socket.socket(fileno=sock_or_fd)
         except OSError as exc:
-            if exc.errno == errno.ENOTSOCK:
+            if exc.errno in (errno.EBADF, errno.ENOTSOCK):
                 raise ValueError(
                     "the file descriptor does not refer to a socket"
                 ) from exc
```