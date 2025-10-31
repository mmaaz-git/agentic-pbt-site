# Bug Report: MultiListener mutates nested MultiListener objects

**Target**: `anyio.streams.stapled.MultiListener`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`MultiListener.__post_init__` destructively mutates nested `MultiListener` objects by calling `del listener.listeners[:]`, leaving them in a broken state with no listeners.

## Property-Based Test

```python
from anyio.streams.stapled import MultiListener
from hypothesis import given, strategies as st


class MockListener:
    async def serve(self, handler, task_group=None):
        pass

    async def aclose(self):
        pass

    @property
    def extra_attributes(self):
        return {}


@given(num_inner=st.integers(min_value=1, max_value=5))
def test_multilistener_preserves_nested_listeners(num_inner):
    inner_listeners = [MockListener() for _ in range(num_inner)]
    inner_multi = MultiListener(inner_listeners)

    outer_multi = MultiListener([MockListener(), inner_multi])

    assert len(inner_multi.listeners) == num_inner
```

**Failing input**: `num_inner=1` (or any positive integer)

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


inner_multi = MultiListener([MockListener(), MockListener()])
print(f"Inner listeners before: {len(inner_multi.listeners)}")

outer_multi = MultiListener([MockListener(), inner_multi])
print(f"Inner listeners after: {len(inner_multi.listeners)}")
```

Output:
```
Inner listeners before: 2
Inner listeners after: 0
```

## Why This Is A Bug

In `stapled.py` lines 111-120, when a nested `MultiListener` is encountered, the code does:

```python
if isinstance(listener, MultiListener):
    listeners.extend(listener.listeners)
    del listener.listeners[:]  # BUG: mutates the input object!
```

This destructively modifies the input `MultiListener` object, which:
1. Violates the principle of non-mutation of inputs
2. Leaves the nested `MultiListener` in a broken state (empty listeners)
3. Makes the nested `MultiListener` unusable after being passed to the constructor
4. Could cause bugs if code tries to reuse the nested `MultiListener`

The type annotation at line 116 even shows awareness this is wrong: `# type: ignore[attr-defined]`

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

The flattening of nested `MultiListener` objects works correctly without mutating the input - the new `MultiListener` simply extends its list with the nested listeners, and there's no need to clear the original.