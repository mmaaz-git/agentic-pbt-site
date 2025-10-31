#!/usr/bin/env python3
import socket

def _validate_socket_fixed(
    sock_or_fd,
    sock_type,
    addr_family=socket.AF_UNSPEC,
    *,
    require_connected=False,
    require_bound=False,
):
    """Simplified version with the bug report's fix"""
    sock = sock_or_fd

    if require_bound:
        try:
            if sock.family in (socket.AF_INET, socket.AF_INET6):
                bound_addr = sock.getsockname()[1]
            else:
                bound_addr = sock.getsockname()
        except OSError:
            bound_addr = None

        # The proposed fix: check for None instead of falsy
        if bound_addr is None:  # Instead of: if not bound_addr
            raise ValueError("the socket must be bound to a local address")

    return sock

# Test the "fixed" version
print("=== Testing with proposed fix ===")

# Test 1: Unbound UNIX socket
sock1 = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
print(f"Unbound UNIX socket getsockname(): {sock1.getsockname()!r}")

try:
    _validate_socket_fixed(sock1, socket.SOCK_DGRAM, require_bound=True)
    print("FIXED VERSION: Accepted unbound UNIX socket (getsockname='') with require_bound=True")
except ValueError as e:
    print(f"FIXED VERSION: Rejected with error: {e}")
sock1.close()

# Test 2: Unbound IPv4 socket
sock2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
print(f"\nUnbound IPv4 socket getsockname(): {sock2.getsockname()}")
print(f"Port value that would be checked: {sock2.getsockname()[1]}")

try:
    _validate_socket_fixed(sock2, socket.SOCK_DGRAM, require_bound=True)
    print("FIXED VERSION: Accepted unbound IPv4 socket with require_bound=True")
except ValueError as e:
    print(f"FIXED VERSION: Rejected with error: {e}")
sock2.close()

# Test 3: Check if OSError case works
print("\n=== Testing OSError case ===")
sock3 = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
sock3.close()  # Close it to trigger OSError

try:
    # This should raise OSError when calling getsockname()
    _validate_socket_fixed(sock3, socket.SOCK_DGRAM, require_bound=True)
    print("FIXED VERSION: Accepted closed socket (unexpected)")
except (ValueError, OSError) as e:
    print(f"FIXED VERSION: Rejected with: {type(e).__name__}: {e}")