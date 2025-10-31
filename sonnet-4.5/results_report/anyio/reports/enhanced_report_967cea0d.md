# Bug Report: anyio.abc._sockets._validate_socket Produces Misleading Error Messages for Invalid File Descriptors

**Target**: `anyio.abc._sockets._validate_socket`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `_validate_socket` function in anyio produces misleading error messages when given an invalid file descriptor with `require_connected=True` or `require_bound=True`. Instead of reporting that the file descriptor is invalid (Bad file descriptor), it incorrectly claims the socket must be connected/bound.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/anyio_env/lib/python3.13/site-packages')

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


if __name__ == "__main__":
    test_validate_socket_error_messages_accurate()
```

<details>

<summary>
**Failing input**: `fd=100`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/envs/anyio_env/lib/python3.13/site-packages/anyio/abc/_sockets.py", line 46, in _validate_socket
    sock = socket.socket(fileno=sock_or_fd)
  File "/home/npc/miniconda/lib/python3.13/socket.py", line 233, in __init__
    _socket.socket.__init__(self, family, type, proto, fileno)
    ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
OSError: [Errno 9] Bad file descriptor

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/56/hypo.py", line 23, in test_validate_socket_error_messages_accurate
    _validate_socket(fd, socket.SOCK_STREAM, require_connected=True)
    ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/pbt/agentic-pbt/envs/anyio_env/lib/python3.13/site-packages/anyio/abc/_sockets.py", line 53, in _validate_socket
    raise ValueError("the socket must be connected") from exc
ValueError: the socket must be connected

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/56/hypo.py", line 35, in <module>
    test_validate_socket_error_messages_accurate()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/56/hypo.py", line 11, in test_validate_socket_error_messages_accurate
    @settings(max_examples=200)
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/anyio_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/56/hypo.py", line 26, in test_validate_socket_error_messages_accurate
    assert "must be connected" not in str(e), (
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Misleading error: got 'the socket must be connected' but real issue is OSError(errno=9)
Falsifying example: test_validate_socket_error_messages_accurate(
    fd=100,
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/anyio_env/lib/python3.13/site-packages')

import socket
from anyio.abc._sockets import _validate_socket

# Test with an invalid file descriptor
invalid_fd = 99999

print("Test 1: Invalid FD with require_connected=True")
print("-" * 50)
try:
    _validate_socket(invalid_fd, socket.SOCK_STREAM, require_connected=True)
except Exception as e:
    print(f"Exception type: {type(e).__name__}")
    print(f"Exception message: {e}")

print("\nTest 2: Invalid FD with require_bound=True")
print("-" * 50)
try:
    _validate_socket(invalid_fd, socket.SOCK_STREAM, require_bound=True)
except Exception as e:
    print(f"Exception type: {type(e).__name__}")
    print(f"Exception message: {e}")

print("\nTest 3: Direct socket creation with invalid FD (for comparison)")
print("-" * 50)
try:
    sock = socket.socket(fileno=invalid_fd)
except Exception as e:
    print(f"Exception type: {type(e).__name__}")
    print(f"Exception message: {e}")
    print(f"Exception errno: {e.errno if hasattr(e, 'errno') else 'N/A'}")
```

<details>

<summary>
Misleading error messages for invalid file descriptors
</summary>
```
Test 1: Invalid FD with require_connected=True
--------------------------------------------------
Exception type: ValueError
Exception message: the socket must be connected

Test 2: Invalid FD with require_bound=True
--------------------------------------------------
Exception type: ValueError
Exception message: the socket must be bound to a local address

Test 3: Direct socket creation with invalid FD (for comparison)
--------------------------------------------------
Exception type: OSError
Exception message: [Errno 9] Bad file descriptor
Exception errno: 9
```
</details>

## Why This Is A Bug

This violates expected behavior by providing misleading diagnostic information when invalid file descriptors are passed to the function. The error handling logic in lines 47-57 of `/home/npc/pbt/agentic-pbt/envs/anyio_env/lib/python3.13/site-packages/anyio/abc/_sockets.py` catches OSError exceptions from `socket.socket(fileno=sock_or_fd)` and incorrectly interprets them based on the `require_connected` and `require_bound` parameters.

When `socket.socket(fileno=invalid_fd)` fails with `OSError(errno=9, "Bad file descriptor")`, the code:
1. Correctly handles `errno.ENOTSOCK` (errno 88) with an accurate error message
2. But for other OSErrors like `errno.EBADF` (errno 9), it assumes the socket exists but isn't in the required state
3. This leads to "the socket must be connected" or "the socket must be bound to a local address" errors when the real issue is that no valid socket exists at all

This misleads developers during debugging - they will try to connect or bind a non-existent socket instead of fixing the invalid file descriptor issue.

## Relevant Context

The `_validate_socket` function is used internally by public API methods across the anyio library:
- `SocketStream.from_socket()` (line 203)
- `UNIXSocketStream.from_socket()` (line 220-222)
- `SocketListener.from_socket()` (line 267)
- `UDPSocket.from_socket()` (line 309)
- `ConnectedUDPSocket.from_socket()` (line 339-343)
- `UNIXDatagramSocket.from_socket()` (line 371)
- `ConnectedUNIXDatagramSocket.from_socket()` (line 402-404)

These public methods accept "a socket object or file descriptor" according to their docstrings, but don't specify behavior for invalid descriptors. While `_validate_socket` is a private function (leading underscore), its error messages bubble up through these public APIs, affecting user-facing error reporting.

The Python standard library's `socket.socket(fileno=...)` constructor raises various OSError exceptions with different error codes:
- `errno.EBADF` (9): Bad file descriptor - when the FD doesn't exist
- `errno.ENOTSOCK` (88): Socket operation on non-socket - when the FD exists but isn't a socket

The current code only properly handles ENOTSOCK but misinterprets EBADF errors.

## Proposed Fix

```diff
--- a/anyio/abc/_sockets.py
+++ b/anyio/abc/_sockets.py
@@ -44,14 +44,11 @@ def _validate_socket(
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
+            # All OSErrors at this point mean the FD is invalid or not a socket
+            # Common cases: EBADF (bad file descriptor), ENOTSOCK (not a socket)
+            raise ValueError(
+                f"invalid file descriptor or not a socket (errno {exc.errno})"
+            ) from exc
     elif isinstance(sock_or_fd, socket.socket):
         sock = sock_or_fd
```