# Bug Report: anyio.abc._sockets._SocketProvider Inconsistent Address Caching

**Target**: `anyio.abc._sockets._SocketProvider.extra_attributes`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_SocketProvider.extra_attributes` property exhibits inconsistent caching for address attributes: `local_address` calls `getsockname()` on every access (dynamic), while `remote_address` captures the value once at initialization (cached). This is the same pattern as the `local_port` vs `remote_port` bug and violates consistency.

## Property-Based Test

```python
from hypothesis import given, strategies as st
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


@given(st.integers(min_value=1, max_value=10))
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
    assert mock.getsockname_calls == initial_getsockname_calls + num_accesses
    assert mock.getpeername_calls == initial_getpeername_calls  # cached!
```

**Failing input**: `num_accesses=2` (or any value ≥ 1)

## Reproducing the Bug

```python
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


class TestProvider(_SocketProvider):
    def __init__(self, sock):
        self._sock = sock

    @property
    def _raw_socket(self):
        return self._sock


mock = MockSocket()
provider = TestProvider(mock)
attrs = provider.extra_attributes

local_addr = attrs[SocketAttribute.local_address]
remote_addr = attrs[SocketAttribute.remote_address]

local_addr()
local_addr()
print(f"getsockname called {mock.getsockname_calls} times")

remote_addr()
remote_addr()
print(f"getpeername called {mock.getpeername_calls} time")
```

## Why This Is A Bug

This is the same inconsistency bug as the `local_port` vs `remote_port` issue:

1. **Inconsistency**: `local_address` and `remote_address` behave differently without justification.

2. **Inefficiency**: `local_address` makes redundant `getsockname()` syscalls on every access.

3. **Pattern violation**: The same socket attributes are cached for remote but not local, which is counterintuitive.

The bug is in `/anyio/abc/_sockets.py` lines 153-165:

```python
attributes: dict[Any, Callable[[], Any]] = {
    SocketAttribute.family: lambda: self._raw_socket.family,
    SocketAttribute.local_address: lambda: convert(
        self._raw_socket.getsockname()  # Calls getsockname() every time
    ),
    SocketAttribute.raw_socket: lambda: self._raw_socket,
}
try:
    peername: tuple[str, int] | None = convert(self._raw_socket.getpeername())
except OSError:
    peername = None

# Provide the remote address for connected sockets
if peername is not None:
    attributes[SocketAttribute.remote_address] = lambda: peername  # Cached value
```

## Fix

Cache `local_address` at initialization, consistent with `remote_address`:

```diff
--- a/anyio/abc/_sockets.py
+++ b/anyio/abc/_sockets.py
@@ -148,16 +148,17 @@ class _SocketProvider(TypedAttributeProvider):
     def extra_attributes(self) -> Mapping[Any, Callable[[], Any]]:
         from .._core._sockets import convert_ipv6_sockaddr as convert

+        local_address = convert(self._raw_socket.getsockname())
         attributes: dict[Any, Callable[[], Any]] = {
             SocketAttribute.family: lambda: self._raw_socket.family,
-            SocketAttribute.local_address: lambda: convert(
-                self._raw_socket.getsockname()
-            ),
+            SocketAttribute.local_address: lambda: local_address,
             SocketAttribute.raw_socket: lambda: self._raw_socket,
         }
         try:
             peername: tuple[str, int] | None = convert(self._raw_socket.getpeername())
```

**Note**: This bug report is related to bug_report_anyio_abc_socket_attribute_caching_2025-09-25_16-30_k7m2.md which describes the same caching inconsistency for `local_port` vs `remote_port`. Both should be fixed together as they represent the same design flaw.