# Bug Report: anyio.abc._sockets._SocketProvider Inconsistent Attribute Caching

**Target**: `anyio.abc._sockets._SocketProvider.extra_attributes`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_SocketProvider.extra_attributes` property exhibits inconsistent caching behavior: `local_port` calls `getsockname()` on every access (dynamic), while `remote_port` captures the value once at initialization (cached). This violates the principle of least surprise and causes unnecessary syscalls.

## Property-Based Test

```python
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

local_port()
local_port()
print(f"getsockname called {mock.getsockname_calls} times")

remote_port()
remote_port()
print(f"getpeername called {mock.getpeername_calls} time")
```

## Why This Is A Bug

1. **Inconsistency**: Similar attributes (`local_port` and `remote_port`) behave differently without justification. This violates the principle of least surprise.

2. **Inefficiency**: `local_port` makes redundant syscalls on every access. For a socket that's accessed frequently, this is wasteful.

3. **Behavioral differences**: The caching strategy affects error handling:
   - If a socket becomes invalid, `local_port` will raise `OSError` on access
   - But `remote_port` will continue returning the cached value

4. **Lack of consistency with similar attributes**: The same pattern exists for `local_address` (dynamic) vs `remote_address` (cached).

The bug is in `/anyio/abc/_sockets.py` lines 168-174:

```python
if self._raw_socket.family in (AddressFamily.AF_INET, AddressFamily.AF_INET6):
    attributes[SocketAttribute.local_port] = (
        lambda: self._raw_socket.getsockname()[1]  # Calls getsockname() every time
    )
    if peername is not None:
        remote_port = peername[1]
        attributes[SocketAttribute.remote_port] = lambda: remote_port  # Cached value
```

## Fix

The fix is to cache `local_port` consistently with `remote_port`:

```diff
--- a/anyio/abc/_sockets.py
+++ b/anyio/abc/_sockets.py
@@ -166,9 +166,10 @@ class _SocketProvider(TypedAttributeProvider):

         # Provide local and remote ports for IP based sockets
         if self._raw_socket.family in (AddressFamily.AF_INET, AddressFamily.AF_INET6):
+            local_port = self._raw_socket.getsockname()[1]
             attributes[SocketAttribute.local_port] = (
-                lambda: self._raw_socket.getsockname()[1]
+                lambda: local_port
             )
             if peername is not None:
                 remote_port = peername[1]
                 attributes[SocketAttribute.remote_port] = lambda: remote_port
```

This change:
- Makes `local_port` behavior consistent with `remote_port`
- Eliminates redundant syscalls
- Maintains the same external behavior for valid sockets
- Is safe because socket addresses don't change after connection for TCP sockets