from hypothesis import given, strategies as st
from unittest.mock import Mock
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
        return ('127.0.0.1', 9090)


@given(st.integers(min_value=1, max_value=10))
def test_local_port_caching_inconsistency(num_accesses):
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
        _ = attrs[SocketAttribute.local_port]()
        _ = attrs[SocketAttribute.remote_port]()

    # Bug: local_port makes num_accesses syscalls, remote_port makes 0
    print(f"num_accesses={num_accesses}: getsockname calls={mock.getsockname_calls - initial_getsockname_calls}, getpeername calls={mock.getpeername_calls - initial_getpeername_calls}")

    assert mock.getsockname_calls == initial_getsockname_calls + num_accesses, f"Expected {initial_getsockname_calls + num_accesses} getsockname calls, got {mock.getsockname_calls}"
    assert mock.getpeername_calls == initial_getpeername_calls, f"Expected {initial_getpeername_calls} getpeername calls (cached), got {mock.getpeername_calls}"


# Run the test
if __name__ == "__main__":
    test_local_port_caching_inconsistency()