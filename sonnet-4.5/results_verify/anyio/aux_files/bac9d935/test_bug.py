#!/usr/bin/env python3

from socket import AddressFamily
from anyio.abc._sockets import _SocketProvider, SocketAttribute


class MockSocket:
    def __init__(self):
        self.family = AddressFamily.AF_INET
        self.getsockname_calls = 0
        self.getpeername_calls = 0

    def getsockname(self):
        self.getsockname_calls += 1
        return ('127.0.0.1', 8080)

    def getpeername(self):
        self.getpeername_calls += 1
        return ('192.168.1.1', 9090)

    def setblocking(self, val):
        pass


class TestProvider(_SocketProvider):
    def __init__(self, sock):
        self._sock = sock

    @property
    def _raw_socket(self):
        return self._sock


def test_address_caching():
    """Test the behavior reported in the bug"""
    mock = MockSocket()
    provider = TestProvider(mock)
    attrs = provider.extra_attributes

    local_addr = attrs[SocketAttribute.local_address]
    remote_addr = attrs[SocketAttribute.remote_address]

    # Reset counters to see behavior from this point
    mock.getsockname_calls = 0
    mock.getpeername_calls = 0

    # First calls
    result1_local = local_addr()
    result1_remote = remote_addr()

    print(f"After first call:")
    print(f"  local_address result: {result1_local}")
    print(f"  remote_address result: {result1_remote}")
    print(f"  getsockname called {mock.getsockname_calls} times")
    print(f"  getpeername called {mock.getpeername_calls} times")

    # Second calls
    result2_local = local_addr()
    result2_remote = remote_addr()

    print(f"\nAfter second call:")
    print(f"  local_address result: {result2_local}")
    print(f"  remote_address result: {result2_remote}")
    print(f"  getsockname called {mock.getsockname_calls} times (total)")
    print(f"  getpeername called {mock.getpeername_calls} times (total)")

    # Check consistency
    print(f"\nConsistency check:")
    print(f"  Local address calls getsockname every time: {mock.getsockname_calls == 2}")
    print(f"  Remote address is cached (no additional calls): {mock.getpeername_calls == 0}")

    return mock.getsockname_calls, mock.getpeername_calls


if __name__ == "__main__":
    print("Testing address caching behavior...")
    print("=" * 50)
    getsockname_calls, getpeername_calls = test_address_caching()
    print("=" * 50)
    print("\nBug report claims:")
    print("  - local_address calls getsockname() every time (not cached)")
    print("  - remote_address is cached and doesn't call getpeername() after init")
    print("\nActual behavior observed:")
    print(f"  - getsockname was called {getsockname_calls} times for 2 accesses")
    print(f"  - getpeername was called {getpeername_calls} times for 2 accesses")

    if getsockname_calls == 2 and getpeername_calls == 0:
        print("\n✓ Bug report is CORRECT: There IS inconsistent caching behavior")
    else:
        print("\n✗ Bug report is INCORRECT: Behavior is different than claimed")