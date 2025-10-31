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
        return ('127.0.0.1', 9090)


class TestProvider(_SocketProvider):
    def __init__(self, sock):
        self._sock = sock

    @property
    def _raw_socket(self):
        return self._sock


mock = MockSocket()
provider = TestProvider(mock)
attrs = provider.extra_attributes

local_port = attrs[SocketAttribute.local_port]
remote_port = attrs[SocketAttribute.remote_port]

print("Initial state:")
print(f"getsockname calls: {mock.getsockname_calls}")
print(f"getpeername calls: {mock.getpeername_calls}")

print("\nCalling local_port() twice:")
local_port()
local_port()
print(f"getsockname called {mock.getsockname_calls} times")

print("\nCalling remote_port() twice:")
remote_port()
remote_port()
print(f"getpeername called {mock.getpeername_calls} time(s)")

print("\nConclusion:")
print(f"local_port calls getsockname() on each access: {mock.getsockname_calls == 2}")
print(f"remote_port uses cached value (no additional calls): {mock.getpeername_calls == 1}")