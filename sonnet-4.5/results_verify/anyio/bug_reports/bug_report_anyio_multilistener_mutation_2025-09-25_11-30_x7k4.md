# Bug Report: anyio.streams.stapled.MultiListener Mutates Nested MultiListeners

**Target**: `anyio.streams.stapled.MultiListener.__post_init__`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When creating a MultiListener from other MultiListener instances, the `__post_init__` method destructively modifies the nested MultiListeners by clearing their internal listeners lists, making them unusable afterwards.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from anyio.streams.stapled import MultiListener

class MockListener:
    def __init__(self, name):
        self.name = name
    async def serve(self, handler, task_group=None):
        pass
    async def aclose(self):
        pass
    @property
    def extra_attributes(self):
        return {}

@given(st.integers(min_value=1, max_value=5))
def test_multi_listener_flattening(n_listeners):
    listeners1 = [MockListener(f"listener_{i}") for i in range(n_listeners)]
    multi1 = MultiListener(listeners1)

    listeners2 = [MockListener(f"listener_{i + n_listeners}") for i in range(n_listeners)]
    multi2 = MultiListener(listeners2)

    combined = MultiListener([multi1, multi2])

    assert len(combined.listeners) == 2 * n_listeners
    assert len(multi1.listeners) == 0
    assert len(multi2.listeners) == 0
```

**Failing input**: Any MultiListener instances passed to another MultiListener

## Reproducing the Bug

```python
import anyio
from anyio.streams.stapled import MultiListener

class MockListener:
    def __init__(self, name):
        self.name = name
    async def serve(self, handler, task_group=None):
        pass
    async def aclose(self):
        pass
    @property
    def extra_attributes(self):
        return {}

async def reproduce():
    listeners1 = [MockListener("A"), MockListener("B")]
    multi1 = MultiListener(listeners1)

    listeners2 = [MockListener("C"), MockListener("D")]
    multi2 = MultiListener(listeners2)

    print(f"multi1 has {len(multi1.listeners)} listeners")
    print(f"multi2 has {len(multi2.listeners)} listeners")

    combined = MultiListener([multi1, multi2])

    print(f"After combining, multi1 has {len(multi1.listeners)} listeners")
    print(f"After combining, multi2 has {len(multi2.listeners)} listeners")

anyio.run(reproduce())
```

## Why This Is A Bug

The `MultiListener.__post_init__` method is designed to flatten nested MultiListener instances by extracting their listeners. However, it does this by destructively modifying the nested MultiListener instances (line 116 in stapled.py):

```python
del listener.listeners[:]
```

This violates the principle of least surprise - a constructor should not modify objects passed to it. After creating a combined MultiListener, the nested MultiListener instances become unusable because their listeners list is empty. If a user tries to use `multi1.serve()` after it's been nested, it will silently do nothing.

This breaks the property that an object should remain in a valid state after being passed to another object's constructor, unless explicitly documented as a move operation (like Rust's move semantics).

## Fix

Instead of modifying the nested MultiListener in place, create a copy of its listeners list:

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

This fix removes the destructive modification while preserving the flattening behavior. The nested MultiListener instances remain usable after being passed to the combined MultiListener.