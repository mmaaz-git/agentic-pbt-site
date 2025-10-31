# Bug Report: anyio.streams MultiListener Mutates Nested MultiListeners

**Target**: `anyio.streams.stapled.MultiListener.__post_init__`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`MultiListener.__post_init__` mutates nested `MultiListener` instances by clearing their `listeners` list, making them unusable after being passed to an outer `MultiListener`. This violates the principle of least surprise and can lead to subtle bugs.

## Property-Based Test

```python
from anyio.streams.stapled import MultiListener


class MockListener:
    def __init__(self, name: str):
        self.name = name

    async def serve(self, handler, task_group=None):
        pass

    async def aclose(self):
        pass

    @property
    def extra_attributes(self):
        return {}


def test_multilistener_should_not_mutate_nested():
    listener1 = MockListener("A")
    listener2 = MockListener("B")

    nested_multi = MultiListener([listener1, listener2])
    original_length = len(nested_multi.listeners)

    outer_multi = MultiListener([nested_multi])

    assert len(nested_multi.listeners) == original_length
```

**Failing input**: Any nested MultiListener configuration

## Reproducing the Bug

```python
from anyio.streams.stapled import MultiListener


class MockListener:
    def __init__(self, name: str):
        self.name = name

    async def serve(self, handler, task_group=None):
        pass

    async def aclose(self):
        pass

    @property
    def extra_attributes(self):
        return {}


listener1 = MockListener("A")
listener2 = MockListener("B")

nested_multi = MultiListener([listener1, listener2])
print(f"Before: nested_multi has {len(nested_multi.listeners)} listeners")

outer_multi = MultiListener([nested_multi])
print(f"After: nested_multi has {len(nested_multi.listeners)} listeners")
print(f"Bug: nested_multi.listeners was cleared!")
```

Output:
```
Before: nested_multi has 2 listeners
After: nested_multi has 0 listeners
Bug: nested_multi.listeners was cleared!
```

## Why This Is A Bug

The current implementation clears the nested `MultiListener`'s listeners list on line 116:

```python
del listener.listeners[:]  # type: ignore[attr-defined]
```

This mutation has several problems:

1. **Violates immutability expectations**: Creating a new MultiListener shouldn't modify existing objects
2. **Makes nested MultiListeners unusable**: After being added to an outer MultiListener, the nested one can no longer serve connections
3. **Surprising behavior**: Users would not expect this side effect
4. **Thread safety issues**: If the nested MultiListener is used elsewhere, this mutation could cause race conditions

## Fix

Instead of mutating the nested MultiListener, simply don't add it to the flattened list:

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

This preserves the flattening behavior while avoiding mutation of the input objects.