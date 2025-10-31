#!/usr/bin/env python3
"""Simple test to reproduce the _validate_socket bug"""

import socket
import errno
import sys

# Add anyio path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/anyio_env/lib/python3.13/site-packages')

from anyio.abc._sockets import _validate_socket

print("Testing _validate_socket with invalid file descriptor...")
print("=" * 60)

# Test with invalid file descriptor (as shown in bug report)
invalid_fd = 999999

# Test with require_connected=True
print("\nTest 1: Invalid FD (999999) with require_connected=True")
try:
    result = _validate_socket(invalid_fd, socket.SOCK_STREAM, require_connected=True)
    print(f"Unexpected success: {result}")
except ValueError as e:
    print(f"ValueError raised: '{e}'")
    print(f"  -> Bug claim: This message is misleading (fd is invalid, not unconnected)")
except OSError as e:
    print(f"OSError raised (errno={e.errno}): {e}")

# Test with require_bound=True
print("\nTest 2: Invalid FD (999999) with require_bound=True")
try:
    result = _validate_socket(invalid_fd, socket.SOCK_STREAM, require_bound=True)
    print(f"Unexpected success: {result}")
except ValueError as e:
    print(f"ValueError raised: '{e}'")
    print(f"  -> Bug claim: This message is misleading (fd is invalid, not unbound)")
except OSError as e:
    print(f"OSError raised (errno={e.errno}): {e}")

# Test with neither require_connected nor require_bound
print("\nTest 3: Invalid FD (999999) with neither require_connected nor require_bound")
try:
    result = _validate_socket(invalid_fd, socket.SOCK_STREAM)
    print(f"Unexpected success: {result}")
except ValueError as e:
    print(f"ValueError raised: {e}")
except OSError as e:
    print(f"OSError raised (errno={e.errno} = {errno.errorcode.get(e.errno, 'UNKNOWN')}): {e}")
    print(f"  -> This correctly propagates the OSError")

# Let's see what socket.socket(fileno=invalid_fd) directly raises
print("\n" + "=" * 60)
print("Direct socket.socket(fileno=999999) test:")
try:
    sock = socket.socket(fileno=999999)
    print(f"Unexpected success: {sock}")
except OSError as e:
    print(f"OSError raised - errno={e.errno} ({errno.errorcode.get(e.errno, 'UNKNOWN')}): {e}")
    print(f"  -> This is EBADF (Bad file descriptor), NOT ENOTSOCK")

# Test with a non-socket file descriptor
print("\n" + "=" * 60)
print("Testing with a regular file descriptor (not a socket):")

with open("/tmp/test_file.txt", "w") as f:
    regular_fd = f.fileno()

    print(f"\nTest 4: Regular file FD ({regular_fd}) with require_connected=True")
    try:
        result = _validate_socket(regular_fd, socket.SOCK_STREAM, require_connected=True)
        print(f"Unexpected success: {result}")
    except ValueError as e:
        print(f"ValueError raised: '{e}'")
        print(f"  -> This correctly identifies non-socket FD")
    except OSError as e:
        print(f"OSError raised (errno={e.errno}): {e}")

    # Direct test on regular file
    print(f"\nDirect socket.socket(fileno={regular_fd}) on regular file:")
    try:
        sock = socket.socket(fileno=regular_fd)
        print(f"Unexpected success: {sock}")
    except OSError as e:
        print(f"OSError raised - errno={e.errno} ({errno.errorcode.get(e.errno, 'UNKNOWN')}): {e}")
        print(f"  -> This is ENOTSOCK as expected for non-socket FD")

import os
if os.path.exists("/tmp/test_file.txt"):
    os.remove("/tmp/test_file.txt")

print("\n" + "=" * 60)
print("SUMMARY:")
print("--------")
print("The bug report claims that when socket.socket(fileno=fd) fails with")
print("EBADF (invalid FD), _validate_socket incorrectly reports:")
print("  - 'the socket must be connected' (if require_connected=True)")
print("  - 'the socket must be bound to a local address' (if require_bound=True)")
print("")
print("The actual error is that the file descriptor is invalid (EBADF),")
print("not that a socket lacks connection/binding.")
print("")
print("The code at lines 52-55 assumes any non-ENOTSOCK OSError means")
print("the socket exists but isn't connected/bound, which is incorrect.")