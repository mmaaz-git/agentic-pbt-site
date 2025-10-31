# Bug Report: anyio.abc._validate_socket Misleading Error Messages

**Target**: `anyio.abc._sockets._validate_socket`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `_validate_socket` function produces misleading error messages when `socket.socket(fileno=fd)` fails with an OSError (other than ENOTSOCK) and either `require_connected=True` or `require_bound=True` is specified. The function incorrectly reports that "the socket must be connected" or "the socket must be bound to a local address" when the actual problem is that the file descriptor is invalid or the socket construction failed for a different reason.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import socket
from anyio.abc._sockets import _validate_socket


@given(
    invalid_fd=st.integers(min_value=10000, max_value=99999),
    require_connected=st.booleans(),
    require_bound=st.booleans(),
)
def test_validate_socket_error_messages_are_accurate(invalid_fd, require_connected, require_bound):
    if not require_connected and not require_bound:
        return

    try:
        _validate_socket(
            invalid_fd,
            socket.SOCK_STREAM,
            require_connected=require_connected,
            require_bound=require_bound,
        )
    except ValueError as e:
        error_msg = str(e)
        assert "must be connected" not in error_msg or not require_connected, \
            f"Error incorrectly claims socket must be connected when fd is invalid: {error_msg}"
        assert "must be bound" not in error_msg or not require_bound, \
            f"Error incorrectly claims socket must be bound when fd is invalid: {error_msg}"
    except OSError:
        pass
```

**Failing input**: Any invalid file descriptor (e.g., `invalid_fd=99999`) with `require_connected=True` or `require_bound=True`

## Reproducing the Bug

```python
import socket
from anyio.abc._sockets import _validate_socket

invalid_fd = 999999

try:
    _validate_socket(invalid_fd, socket.SOCK_STREAM, require_connected=True)
except ValueError as e:
    print(f"Error message: {e}")

try:
    _validate_socket(invalid_fd, socket.SOCK_STREAM, require_bound=True)
except ValueError as e:
    print(f"Error message: {e}")
```

Output:
```
Error message: the socket must be connected
Error message: the socket must be bound to a local address
```

Both error messages are misleading because the real issue is that file descriptor 999999 is invalid, not that a socket isn't connected or bound.

## Why This Is A Bug

Lines 47-57 in `_sockets.py` handle OSError exceptions from `socket.socket(fileno=sock_or_fd)`:

```python
except OSError as exc:
    if exc.errno == errno.ENOTSOCK:
        raise ValueError("the file descriptor does not refer to a socket") from exc
    elif require_connected:
        raise ValueError("the socket must be connected") from exc
    elif require_bound:
        raise ValueError("the socket must be bound to a local address") from exc
    else:
        raise
```

The logic assumes that if `socket.socket(fileno=fd)` raises an OSError that is NOT `errno.ENOTSOCK`, and `require_connected` or `require_bound` is True, then the error must be because the socket isn't connected or bound. This assumption is incorrect. The `socket.socket(fileno=fd)` constructor can fail for many reasons unrelated to connection or binding status:
- Invalid file descriptor (EBADF)
- File descriptor refers to a closed socket
- Permission issues
- Other OS-level errors

The error messages mislead developers into thinking their socket configuration is wrong, when the actual problem is often that they're using an invalid file descriptor.

## Fix

The `require_connected` and `require_bound` checks should only apply to validation that happens AFTER the socket is successfully constructed. The OSError from the socket constructor should be re-raised or wrapped with a more accurate message:

```diff
--- a/anyio/abc/_sockets.py
+++ b/anyio/abc/_sockets.py
@@ -48,10 +48,6 @@ def _validate_socket(
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

This fix removes the misleading error messages and lets the original OSError propagate (or fall through to the else: raise), providing more accurate information about what actually went wrong.