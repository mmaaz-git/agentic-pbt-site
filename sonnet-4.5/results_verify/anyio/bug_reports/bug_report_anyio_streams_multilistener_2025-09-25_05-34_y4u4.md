# Bug Report: anyio.streams.stapled.MultiListener Mutates Input

**Target**: `anyio.streams.stapled.MultiListener`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Creating a `MultiListener` with another `MultiListener` as input unexpectedly empties the input `MultiListener`'s listeners list, violating the principle that object construction should not mutate existing objects.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from anyio.streams.stapled import MultiListener
from dataclasses import dataclass


@dataclass
class MockListener:
    name: str

    async def serve(self, handler, task_group=None):
        pass

    async def aclose(self):
        pass

    @property
    def extra_attributes(self):
        return {}


@given(st.integers(min_value=1, max_value=10))
def test_multilistener_does_not_mutate_input(num_listeners):
    listeners = [MockListener(f"listener{i}") for i in range(num_listeners)]
    multi1 = MultiListener(listeners=listeners)

    original_count = len(multi1.listeners)
    assert original_count == num_listeners

    multi2 = MultiListener(listeners=[multi1])

    assert len(multi1.listeners) == original_count
```

**Failing input**: `num_listeners=1`

## Reproducing the Bug

```python
from anyio.streams.stapled import MultiListener
from dataclasses import dataclass


@dataclass
class MockListener:
    name: str

    async def serve(self, handler, task_group=None):
        pass

    async def aclose(self):
        pass

    @property
    def extra_attributes(self):
        return {}


listener1 = MockListener("listener1")
listener2 = MockListener("listener2")

multi1 = MultiListener(listeners=[listener1, listener2])
print(f"multi1.listeners: {len(multi1.listeners)}")

multi2 = MultiListener(listeners=[multi1])
print(f"multi1.listeners after creating multi2: {len(multi1.listeners)}")

assert len(multi1.listeners) == 0
```

## Why This Is A Bug

When creating `multi2` with `multi1` as input, the `__post_init__` method modifies `multi1.listeners` by calling `del listener.listeners[:]` (line 116 in stapled.py). This violates the expected behavior that constructing a new object should not mutate existing objects. This could cause unexpected behavior if users try to reuse `multi1` after passing it to another `MultiListener`.

## Fix

```diff
--- a/anyio/streams/stapled.py
+++ b/anyio/streams/stapled.py
@@ -113,7 +113,6 @@ class MultiListener(Generic[T_Stream], Listener[T_Stream]):
         for listener in self.listeners:
             if isinstance(listener, MultiListener):
                 listeners.extend(listener.listeners)
-                del listener.listeners[:]  # type: ignore[attr-defined]
             else:
                 listeners.append(listener)

```
