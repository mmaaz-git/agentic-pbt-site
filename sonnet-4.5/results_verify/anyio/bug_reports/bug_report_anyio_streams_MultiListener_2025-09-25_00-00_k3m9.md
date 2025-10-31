# Bug Report: anyio.streams.stapled.MultiListener Mutates Input

**Target**: `anyio.streams.stapled.MultiListener.__post_init__`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When constructing a `MultiListener` with another `MultiListener` as input, the constructor mutates the input `MultiListener` by clearing its `listeners` list, leaving the original object in an unusable state.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
from anyio.streams.stapled import MultiListener


class MockListener:
    async def serve(self, handler, task_group=None):
        pass

    async def aclose(self):
        pass

    @property
    def extra_attributes(self):
        return {}


@given(st.integers(min_value=1, max_value=10))
@settings(max_examples=500)
def test_multilistener_doesnt_mutate_input(num_listeners):
    async def test():
        inner_listeners = [MockListener() for _ in range(num_listeners)]
        inner_multi = MultiListener(listeners=inner_listeners)

        original_count = len(inner_multi.listeners)

        outer_multi = MultiListener(listeners=[inner_multi])

        after_count = len(inner_multi.listeners)
        assert after_count == original_count, f"Creating outer MultiListener mutated inner: {original_count} -> {after_count}"

    anyio.run(test)
```

**Failing input**: `num_listeners=1` (or any positive integer)

## Reproducing the Bug

```python
from anyio.streams.stapled import MultiListener


class MockListener:
    async def serve(self, handler, task_group=None):
        pass

    async def aclose(self):
        pass

    @property
    def extra_attributes(self):
        return {}


inner_listeners = [MockListener()]
inner_multi = MultiListener(listeners=inner_listeners)

print(f"Before: len(inner_multi.listeners) = {len(inner_multi.listeners)}")

outer_multi = MultiListener(listeners=[inner_multi])

print(f"After: len(inner_multi.listeners) = {len(inner_multi.listeners)}")
```

**Expected output**: Both print statements show `1`
**Actual output**: First shows `1`, second shows `0`

## Why This Is A Bug

The `MultiListener.__post_init__` method contains this code (line 116):

```python
if isinstance(listener, MultiListener):
    listeners.extend(listener.listeners)
    del listener.listeners[:]  # type: ignore[attr-defined]
```

The line `del listener.listeners[:]` mutates the input object by clearing its listeners list. This violates fundamental principles:

1. **Constructors should not mutate inputs**: Objects passed to constructors should remain unchanged
2. **Principle of least surprise**: Users don't expect object construction to modify other objects
3. **Reusability**: After passing an inner `MultiListener` to an outer one, the inner object becomes unusable (has no listeners)

This would cause real bugs if code tries to reuse or inspect a `MultiListener` after it has been passed to another `MultiListener`.

## Fix

Remove the line that clears the input's listeners list. The flattening can be done without mutation:

```diff
--- a/anyio/streams/stapled.py
+++ b/anyio/streams/stapled.py
@@ -111,9 +111,8 @@ class MultiListener(Generic[T_Stream], Listener[T_Stream]):
     def __post_init__(self) -> None:
         listeners: list[Listener[T_Stream]] = []
         for listener in self.listeners:
             if isinstance(listener, MultiListener):
                 listeners.extend(listener.listeners)
-                del listener.listeners[:]  # type: ignore[attr-defined]
             else:
                 listeners.append(listener)

         self.listeners = listeners
```

This change maintains the same flattening behavior without mutating the input `MultiListener` objects.