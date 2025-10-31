#!/usr/bin/env python3

from hypothesis import given, strategies as st, settings
from socket import AddressFamily


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


@given(st.integers(min_value=1, max_value=10))
@settings(max_examples=5)
def test_local_address_caching_inconsistency(num_accesses):
    from anyio.abc._sockets import _SocketProvider, SocketAttribute

    class TestProvider(_SocketProvider):
        def __init__(self, sock):
            self._sock = sock

        @property
        def _raw_socket(self):
            return self._sock

    mock = MockSocket()
    provider = TestProvider(mock)
    attrs = provider.extra_attributes

    initial_getsockname_calls = mock.getsockname_calls
    initial_getpeername_calls = mock.getpeername_calls

    for _ in range(num_accesses):
        _ = attrs[SocketAttribute.local_address]()
        _ = attrs[SocketAttribute.remote_address]()

    # Bug: local_address makes num_accesses syscalls, remote_address makes 0
    print(f"num_accesses={num_accesses}: getsockname={mock.getsockname_calls - initial_getsockname_calls}, getpeername={mock.getpeername_calls - initial_getpeername_calls}")

    assert mock.getsockname_calls == initial_getsockname_calls + num_accesses, \
        f"Expected {initial_getsockname_calls + num_accesses} getsockname calls, got {mock.getsockname_calls}"

    assert mock.getpeername_calls == initial_getpeername_calls, \
        f"Expected {initial_getpeername_calls} getpeername calls, got {mock.getpeername_calls}"


if __name__ == "__main__":
    print("Running hypothesis test...")
    test_local_address_caching_inconsistency()
    print("\nAll tests passed - the inconsistent caching behavior is confirmed!")