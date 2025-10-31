# Bug Report: anyio.streams.stapled.MultiListener Destructive Mutation

**Target**: `anyio.streams.stapled.MultiListener`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`MultiListener.__post_init__` destructively modifies nested `MultiListener` objects by clearing their listeners, causing unexpected side effects when creating new MultiListener instances.

## Property-Based Test

```python
from anyio.streams.stapled import MultiListener
from hypothesis import given, strategies as st


class MockListener:
    def __init__(self, name):
        self.name = name

    @property
    def extra_attributes(self):
        return {}


@given(num_listeners=st.integers(min_value=1, max_value=5))
def test_multilistener_does_not_mutate_nested(num_listeners):
    listeners = [MockListener(f"listener_{i}") for i in range(num_listeners)]
    nested = MultiListener(listeners=listeners)

    original_count = len(nested.listeners)
    assert original_count == num_listeners

    flat = MultiListener(listeners=[nested])

    assert len(nested.listeners) == original_count
```

**Failing input**: `num_listeners=1`

## Reproducing the Bug

```python
from anyio.streams.stapled import MultiListener


class MockListener:
    @property
    def extra_attributes(self):
        return {}


listener = MockListener()
nested = MultiListener(listeners=[listener])
print(f"Before: nested.listeners = {nested.listeners}")

flat = MultiListener(listeners=[nested])
print(f"After: nested.listeners = {nested.listeners}")
```

Output:
```
Before: nested.listeners = [<__main__.MockListener object at 0x...>]
After: nested.listeners = []
```

## Why This Is A Bug

The `MultiListener.__post_init__` method contains this code at `anyio/streams/stapled.py:118`:

```python
if isinstance(listener, MultiListener):
    listeners.extend(listener.listeners)
    del listener.listeners[:]  # type: ignore[attr-defined]
```

This destructively clears the `listeners` attribute of the nested `MultiListener` object. Creating a new object should never mutate existing objects as a side effect. This violates the principle of immutability and least surprise, and makes `MultiListener` objects unsafe to reuse after passing them to another `MultiListener`.

## Fix

```diff
--- a/anyio/streams/stapled.py
+++ b/anyio/streams/stapled.py
@@ -115,7 +115,6 @@ class MultiListener(Generic[T_Stream], Listener[T_Stream]):
         for listener in self.listeners:
             if isinstance(listener, MultiListener):
                 listeners.extend(listener.listeners)
-                del listener.listeners[:]  # type: ignore[attr-defined]
             else:
                 listeners.append(listener)
```

The deletion is unnecessary - the flattening still works correctly by copying the nested listeners into the new list, without needing to mutate the original.