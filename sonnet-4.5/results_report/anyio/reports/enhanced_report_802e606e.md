# Bug Report: anyio.streams.stapled.MultiListener Destructive Mutation of Nested Instances

**Target**: `anyio.streams.stapled.MultiListener`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `MultiListener.__post_init__` method destructively clears the `listeners` list of any nested `MultiListener` objects passed to its constructor, making the original MultiListener instances unusable after being passed as arguments.

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


# Run the test
if __name__ == "__main__":
    test_multilistener_does_not_mutate_nested()
```

<details>

<summary>
**Failing input**: `num_listeners=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/1/hypo.py", line 29, in <module>
    test_multilistener_does_not_mutate_nested()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/1/hypo.py", line 15, in test_multilistener_does_not_mutate_nested
    def test_multilistener_does_not_mutate_nested(num_listeners):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/1/hypo.py", line 24, in test_multilistener_does_not_mutate_nested
    assert len(nested.listeners) == original_count
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_multilistener_does_not_mutate_nested(
    num_listeners=1,
)
```
</details>

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

<details>

<summary>
Output shows nested.listeners is emptied after creating flat MultiListener
</summary>
```
Before: nested.listeners = [<__main__.MockListener object at 0x7ab062c67230>]
After: nested.listeners = []
```
</details>

## Why This Is A Bug

This behavior violates fundamental Python programming conventions where constructors should not destructively mutate their arguments unless explicitly documented with strong warnings. The issue stems from line 116 in `anyio/streams/stapled.py`:

```python
del listener.listeners[:]  # type: ignore[attr-defined]
```

While the class documentation mentions that nested MultiListeners will have their listeners "moved" into the parent, this terminology is ambiguous and doesn't clearly communicate that the source MultiListener will be rendered unusable. In Python, "moving" typically implies transferring references, not destructive mutation. Most Python developers would expect behavior similar to `list.extend()` which copies references without mutating the source.

The presence of `# type: ignore[attr-defined]` suggests the developers had to bypass type checking to implement this mutation, which is often indicative of problematic design. This makes MultiListener objects unexpectedly stateful and unsafe to reuse after passing them to another MultiListener constructor.

## Relevant Context

The MultiListener class is designed to combine multiple network listeners into one, allowing servers to accept connections from multiple sources simultaneously. The flattening behavior (converting nested MultiListeners into a flat list) is intentional and useful, but the destructive mutation is unnecessary for achieving this functionality.

The bug particularly affects scenarios where:
- Users want to create multiple MultiListener configurations that share some common listeners
- Testing code that needs to verify MultiListener behavior before and after composition
- Any code that assumes Python objects are not mutated by constructors

Source code location: `/home/npc/pbt/agentic-pbt/envs/anyio_env/lib/python3.13/site-packages/anyio/streams/stapled.py:116`

## Proposed Fix

The destructive mutation can be removed without affecting the flattening functionality. The listeners are already being copied into the new list via `extend()`, making the deletion unnecessary:

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