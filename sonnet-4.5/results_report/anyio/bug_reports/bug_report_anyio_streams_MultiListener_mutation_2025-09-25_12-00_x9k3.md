# Bug Report: anyio.streams.stapled.MultiListener Input Mutation

**Target**: `anyio.streams.stapled.MultiListener`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`MultiListener.__post_init__` mutates nested `MultiListener` objects passed as input by clearing their `listeners` list, violating the principle of immutability for constructor arguments.

## Property-Based Test

```python
from hypothesis import given, strategies as st
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
def test_multilistener_does_not_mutate_inputs(num_listeners: int):
    listeners = [MockListener() for _ in range(num_listeners)]
    nested = MultiListener(listeners=listeners)
    original_count = len(nested.listeners)

    MultiListener(listeners=[nested, MockListener()])

    assert len(nested.listeners) == original_count, \
        f"MultiListener mutated input: expected {original_count} listeners, but found {len(nested.listeners)}"
```

**Failing input**: Any positive integer (e.g., `2`)

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


nested_multi = MultiListener(listeners=[MockListener(), MockListener()])
print(f"Before: {len(nested_multi.listeners)} listeners")

outer_multi = MultiListener(listeners=[nested_multi, MockListener()])
print(f"After: {len(nested_multi.listeners)} listeners")
```

Expected: Both print statements show 2 listeners
Actual: First prints 2, second prints 0

## Why This Is A Bug

The `MultiListener.__post_init__` method in `/lib/python3.13/site-packages/anyio/streams/stapled.py` lines 111-120 contains:

```python
def __post_init__(self) -> None:
    listeners: list[Listener[T_Stream]] = []
    for listener in self.listeners:
        if isinstance(listener, MultiListener):
            listeners.extend(listener.listeners)
            del listener.listeners[:]  # type: ignore[attr-defined]
        else:
            listeners.append(listener)

    self.listeners = listeners
```

Line 116 (`del listener.listeners[:]`) **mutates the input object** by clearing its listeners list. This violates the expected behavior that constructor arguments should not be modified. When a user creates a `MultiListener` that contains another `MultiListener`, the nested one becomes unusable (empty) after construction.

This is a violation of the principle of least surprise - constructors should not have side effects on their arguments.

## Fix

Remove the mutation by not clearing the nested MultiListener's list:

```diff
 def __post_init__(self) -> None:
     listeners: list[Listener[T_Stream]] = []
     for listener in self.listeners:
         if isinstance(listener, MultiListener):
             listeners.extend(listener.listeners)
-            del listener.listeners[:]  # type: ignore[attr-defined]
         else:
             listeners.append(listener)

     self.listeners = listeners
```

The deletion appears to have been an attempt to prevent the nested MultiListener from being used after flattening, but this is overly aggressive and breaks user code. The nested MultiListener should remain intact.