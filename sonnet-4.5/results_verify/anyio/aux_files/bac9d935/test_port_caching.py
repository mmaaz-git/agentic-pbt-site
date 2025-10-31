#!/usr/bin/env python3

from socket import AddressFamily
from anyio.abc._sockets import _SocketProvider, SocketAttribute


class MockSocket:
    def __init__(self):
        self.family = AddressFamily.AF_INET  # Use IPv4 to enable port attributes
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


def test_port_caching():
    """Test port attribute caching behavior"""
    mock = MockSocket()
    provider = TestProvider(mock)
    attrs = provider.extra_attributes

    # Check if port attributes exist
    if SocketAttribute.local_port in attrs and SocketAttribute.remote_port in attrs:
        local_port_fn = attrs[SocketAttribute.local_port]
        remote_port_fn = attrs[SocketAttribute.remote_port]

        # Reset counters
        mock.getsockname_calls = 0
        mock.getpeername_calls = 0

        # First calls
        port1_local = local_port_fn()
        port1_remote = remote_port_fn()

        print(f"After first call:")
        print(f"  local_port result: {port1_local}")
        print(f"  remote_port result: {port1_remote}")
        print(f"  getsockname called {mock.getsockname_calls} times")
        print(f"  getpeername called {mock.getpeername_calls} times")

        # Second calls
        port2_local = local_port_fn()
        port2_remote = remote_port_fn()

        print(f"\nAfter second call:")
        print(f"  local_port result: {port2_local}")
        print(f"  remote_port result: {port2_remote}")
        print(f"  getsockname called {mock.getsockname_calls} times (total)")
        print(f"  getpeername called {mock.getpeername_calls} times (total)")

        print(f"\nPort caching behavior:")
        print(f"  Local port calls getsockname every time: {mock.getsockname_calls == 2}")
        print(f"  Remote port is cached (no additional calls): {mock.getpeername_calls == 0}")

        return mock.getsockname_calls, mock.getpeername_calls
    else:
        print("Port attributes not available - this only happens for non-connected sockets")
        return None, None


if __name__ == "__main__":
    print("Testing port caching behavior...")
    print("=" * 50)
    getsockname_calls, getpeername_calls = test_port_caching()
    print("=" * 50)

    if getsockname_calls is not None:
        if getsockname_calls == 2 and getpeername_calls == 0:
            print("\n✓ Similar inconsistent caching for ports: local_port not cached, remote_port cached")
        else:
            print("\n✗ Port caching behavior is different than expected")