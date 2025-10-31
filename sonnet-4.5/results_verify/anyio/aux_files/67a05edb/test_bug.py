#!/usr/bin/env python3
"""Test script to reproduce the reported bug in _validate_socket"""

import socket
import errno
import tempfile
import sys
import os

# First, let's copy the _validate_socket function from the bug report
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


# Test 1: File descriptor (should raise ValueError according to bug report)
print("Test 1: File descriptor for non-socket")
with tempfile.NamedTemporaryFile() as f:
    try:
        _validate_socket(f.fileno(), socket.SOCK_STREAM)
    except ValueError as e:
        print(f"  File FD: Raises ValueError ✓ - {e}")
    except OSError as e:
        print(f"  File FD: Raises OSError ✗ - errno={e.errno}, msg={e}")

# Test 2: Invalid file descriptor (bug report claims this raises OSError instead of ValueError)
print("\nTest 2: Invalid file descriptor")
try:
    _validate_socket(999999, socket.SOCK_STREAM)
except ValueError as e:
    print(f"  Invalid FD: Raises ValueError ✓ - {e}")
except OSError as e:
    print(f"  Invalid FD: Raises OSError ✗ - errno={e.errno} (EBADF={errno.EBADF}), msg={e}")

# Test 3: Let's test with a few other invalid FDs
print("\nTest 3: Multiple invalid FDs")
for fd in [10000, 50000, 999999]:
    try:
        _validate_socket(fd, socket.SOCK_STREAM)
    except ValueError as e:
        print(f"  FD {fd}: Raises ValueError - {e}")
    except OSError as e:
        print(f"  FD {fd}: Raises OSError - errno={e.errno} (EBADF={errno.EBADF})")

# Test 4: Test with a closed socket's FD
print("\nTest 4: Closed socket FD")
s = socket.socket()
fd = s.fileno()
s.close()
try:
    _validate_socket(fd, socket.SOCK_STREAM)
except ValueError as e:
    print(f"  Closed socket FD: Raises ValueError - {e}")
except OSError as e:
    print(f"  Closed socket FD: Raises OSError - errno={e.errno}")

# Test 5: Run the property-based test from the bug report
print("\nTest 5: Property-based test simulation")

def test_validate_socket_invalid_fd_raises_valueerror(fd):
    """The property test from the bug report"""
    # Skip if FD happens to be valid (unlikely but possible)
    try:
        os.fstat(fd)
        return None  # Skip this FD if it's actually valid
    except OSError:
        pass  # This is what we expect - an invalid FD

    # Now test the function
    try:
        _validate_socket(fd, socket.SOCK_STREAM)
        assert False, f"Expected exception for FD {fd}"
    except ValueError:
        # Bug report expects this
        return True
    except OSError as e:
        # But we actually get this
        assert e.errno == errno.EBADF, f"Unexpected errno: {e.errno}"
        return False

# Run a few samples
print("Running manual test samples...")
passed = 0
failed = 0
skipped = 0
for fd in [10000, 20000, 50000, 100000, 999999]:
    try:
        result = test_validate_socket_invalid_fd_raises_valueerror(fd)
        if result is True:
            passed += 1
        elif result is False:
            failed += 1
        elif result is None:
            skipped += 1
    except AssertionError:
        failed += 1

print(f"  Passed (raised ValueError): {passed}")
print(f"  Failed (raised OSError): {failed}")
print(f"  Skipped (valid FD): {skipped}")