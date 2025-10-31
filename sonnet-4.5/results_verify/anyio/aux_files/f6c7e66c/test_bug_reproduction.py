#!/usr/bin/env python3
"""Test to reproduce the _validate_socket bug"""

import socket
import errno
import sys
import os

# Add anyio path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/anyio_env/lib/python3.13/site-packages')

from anyio.abc._sockets import _validate_socket

print("Testing _validate_socket with invalid file descriptor...")
print("=" * 60)

# Test with invalid file descriptor
invalid_fd = 999999

# Test with require_connected=True
print("\nTest 1: Invalid FD with require_connected=True")
try:
    result = _validate_socket(invalid_fd, socket.SOCK_STREAM, require_connected=True)
    print(f"Unexpected success: {result}")
except ValueError as e:
    print(f"ValueError raised: {e}")
except OSError as e:
    print(f"OSError raised (errno={e.errno}): {e}")

# Test with require_bound=True
print("\nTest 2: Invalid FD with require_bound=True")
try:
    result = _validate_socket(invalid_fd, socket.SOCK_STREAM, require_bound=True)
    print(f"Unexpected success: {result}")
except ValueError as e:
    print(f"ValueError raised: {e}")
except OSError as e:
    print(f"OSError raised (errno={e.errno}): {e}")

# Test with neither require_connected nor require_bound
print("\nTest 3: Invalid FD with neither require_connected nor require_bound")
try:
    result = _validate_socket(invalid_fd, socket.SOCK_STREAM)
    print(f"Unexpected success: {result}")
except ValueError as e:
    print(f"ValueError raised: {e}")
except OSError as e:
    print(f"OSError raised (errno={e.errno}): {e}")

# Let's also test what kind of OSError we get
print("\n" + "=" * 60)
print("Direct socket.socket(fileno=invalid_fd) test:")
try:
    sock = socket.socket(fileno=invalid_fd)
    print(f"Unexpected success: {sock}")
except OSError as e:
    print(f"OSError raised - errno={e.errno} ({errno.errorcode.get(e.errno, 'UNKNOWN')}): {e}")

# Test with a non-socket file descriptor
print("\n" + "=" * 60)
print("Testing with a regular file descriptor (not a socket):")

# Create a regular file
with open("/tmp/test_file.txt", "w") as f:
    regular_fd = f.fileno()

    print(f"\nTest 4: Regular file FD ({regular_fd}) with require_connected=True")
    try:
        result = _validate_socket(regular_fd, socket.SOCK_STREAM, require_connected=True)
        print(f"Unexpected success: {result}")
    except ValueError as e:
        print(f"ValueError raised: {e}")
    except OSError as e:
        print(f"OSError raised (errno={e.errno}): {e}")

# Clean up
if os.path.exists("/tmp/test_file.txt"):
    os.remove("/tmp/test_file.txt")

print("\n" + "=" * 60)
print("Testing the hypothesis test case...")

# The hypothesis test
from hypothesis import given, strategies as st

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
        # This is acceptable - OSError means the FD was invalid
        pass

# Run a few specific test cases
test_cases = [
    (99999, True, False),  # invalid fd with require_connected
    (99999, False, True),  # invalid fd with require_bound
    (99999, True, True),   # invalid fd with both
    (10001, True, False),
    (50000, False, True),
]

for fd, req_conn, req_bound in test_cases:
    print(f"\nTesting fd={fd}, require_connected={req_conn}, require_bound={req_bound}")
    try:
        test_validate_socket_error_messages_are_accurate(fd, req_conn, req_bound)
        print("  Test passed")
    except AssertionError as e:
        print(f"  Test FAILED: {e}")