# Bug Report: anyio.abc._sockets._SocketProvider Inconsistent Attribute Caching Behavior

**Target**: `anyio.abc._sockets._SocketProvider.extra_attributes`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_SocketProvider.extra_attributes` property exhibits inconsistent caching behavior where `local_port` makes a system call on every access while `remote_port` caches its value at initialization, causing unnecessary syscalls and violating the principle of least surprise.

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

<details>

<summary>
**Failing input**: `num_accesses=1` (or any value ≥ 1)
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/55
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_local_port_caching_inconsistency PASSED

============================== 1 passed in 0.03s ===============================
```
</details>

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

print("Before any calls:")
print(f"  getsockname called {mock.getsockname_calls} times")
print(f"  getpeername called {mock.getpeername_calls} times")

print("\nFirst call to local_port():")
result1 = local_port()
print(f"  Result: {result1}")
print(f"  getsockname called {mock.getsockname_calls} times (total)")

print("\nSecond call to local_port():")
result2 = local_port()
print(f"  Result: {result2}")
print(f"  getsockname called {mock.getsockname_calls} times (total)")

print("\nFirst call to remote_port():")
result3 = remote_port()
print(f"  Result: {result3}")
print(f"  getpeername called {mock.getpeername_calls} times (total)")

print("\nSecond call to remote_port():")
result4 = remote_port()
print(f"  Result: {result4}")
print(f"  getpeername called {mock.getpeername_calls} times (total)")

print("\n=== SUMMARY ===")
print(f"local_port() was called 2 times, getsockname() was called {mock.getsockname_calls} times")
print(f"remote_port() was called 2 times, getpeername() was called {mock.getpeername_calls} time(s)")
print("\nThis demonstrates the inconsistent caching:")
print("- local_port calls getsockname() on EVERY access (no caching)")
print("- remote_port calls getpeername() ONCE during initialization (cached)")
```

<details>

<summary>
Output showing inconsistent syscall behavior
</summary>
```
Before any calls:
  getsockname called 0 times
  getpeername called 1 times

First call to local_port():
  Result: 8080
  getsockname called 1 times (total)

Second call to local_port():
  Result: 8080
  getsockname called 2 times (total)

First call to remote_port():
  Result: 9090
  getpeername called 1 times (total)

Second call to remote_port():
  Result: 9090
  getpeername called 1 times (total)

=== SUMMARY ===
local_port() was called 2 times, getsockname() was called 2 times
remote_port() was called 2 times, getpeername() was called 1 time(s)

This demonstrates the inconsistent caching:
- local_port calls getsockname() on EVERY access (no caching)
- remote_port calls getpeername() ONCE during initialization (cached)
```
</details>

## Why This Is A Bug

This violates expected behavior in several ways:

1. **Inconsistent behavior between similar attributes**: The `local_port` and `remote_port` attributes are semantically similar - both represent port numbers of a socket connection. Users would reasonably expect them to behave consistently, yet one makes syscalls on every access while the other caches its value.

2. **Performance inefficiency**: Each access to `local_port` triggers a `getsockname()` system call. For applications that frequently check socket attributes, this creates unnecessary overhead. System calls have significant overhead compared to returning a cached value.

3. **Violates the principle of least surprise**: The documentation for `SocketAttribute.local_port` and `SocketAttribute.remote_port` doesn't indicate any behavioral difference between them. Both are simply documented as returning integer port values for IP-based sockets.

4. **Error handling inconsistency**: If a socket becomes invalid after initialization:
   - `local_port` will raise an `OSError` when accessed (due to the fresh syscall)
   - `remote_port` will continue returning the cached value from initialization
   This creates different failure modes for what should be similar operations.

5. **No technical justification**: For connected TCP sockets, both the local and remote addresses are fixed after connection establishment. There's no technical reason why one should be dynamic while the other is cached.

## Relevant Context

The bug is located in `/anyio/abc/_sockets.py` at lines 168-174. The code shows the clear difference in implementation:

```python
if self._raw_socket.family in (AddressFamily.AF_INET, AddressFamily.AF_INET6):
    attributes[SocketAttribute.local_port] = (
        lambda: self._raw_socket.getsockname()[1]  # Dynamic - calls getsockname() every time
    )
    if peername is not None:
        remote_port = peername[1]
        attributes[SocketAttribute.remote_port] = lambda: remote_port  # Cached - uses captured value
```

The `remote_port` is cached because the code already has the `peername` value from calling `getpeername()` earlier (line 159). However, `local_port` directly calls `getsockname()` in its lambda, making it dynamic.

This same pattern exists for the address attributes as well:
- `local_address` (line 157): `lambda: self._raw_socket.getsockname()` - Dynamic
- `remote_address` (line 165): `lambda: peername` - Cached

The anyio documentation doesn't specify any caching behavior, so this inconsistency appears to be an implementation oversight rather than intentional design.

## Proposed Fix

```diff
--- a/anyio/abc/_sockets.py
+++ b/anyio/abc/_sockets.py
@@ -154,10 +154,11 @@ class _SocketProvider(TypedAttributeProvider):
         attributes: dict[Any, Callable[[], Any]] = {}
         self._cached_properties = attributes
         attributes[SocketAttribute.family] = lambda: self._raw_socket.family
+        local_address = self._raw_socket.getsockname()
         attributes[SocketAttribute.local_address] = (
-            lambda: self._raw_socket.getsockname()
+            lambda: local_address
         )
         try:
             peername = self._raw_socket.getpeername()
         except OSError:
@@ -166,9 +167,10 @@ class _SocketProvider(TypedAttributeProvider):
         # Provide the remote address for connected sockets
         if peername is not None:
             attributes[SocketAttribute.remote_address] = lambda: peername

         # Provide local and remote ports for IP based sockets
         if self._raw_socket.family in (AddressFamily.AF_INET, AddressFamily.AF_INET6):
+            local_port = local_address[1]
             attributes[SocketAttribute.local_port] = (
-                lambda: self._raw_socket.getsockname()[1]
+                lambda: local_port
             )
             if peername is not None:
                 remote_port = peername[1]
```