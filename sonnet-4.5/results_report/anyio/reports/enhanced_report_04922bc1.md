# Bug Report: anyio.streams.stapled.MultiListener Destructively Mutates Nested MultiListeners

**Target**: `anyio.streams.stapled.MultiListener.__post_init__`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

MultiListener's constructor destructively mutates any nested MultiListener objects by clearing their internal listeners list, rendering them unusable after being passed to another MultiListener's constructor.

## Property-Based Test

```python
import hypothesis
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


@hypothesis.given(hypothesis.strategies.just(None))
def test_multilistener_should_not_mutate_nested(dummy):
    listener1 = MockListener("A")
    listener2 = MockListener("B")

    nested_multi = MultiListener([listener1, listener2])
    original_length = len(nested_multi.listeners)

    outer_multi = MultiListener([nested_multi])

    assert len(nested_multi.listeners) == original_length, \
        f"Expected {original_length} listeners, but found {len(nested_multi.listeners)}"


if __name__ == "__main__":
    test_multilistener_should_not_mutate_nested()
```

<details>

<summary>
**Failing input**: `dummy=None`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/63/hypo.py", line 35, in <module>
    test_multilistener_should_not_mutate_nested()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/63/hypo.py", line 21, in test_multilistener_should_not_mutate_nested
    def test_multilistener_should_not_mutate_nested(dummy):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/63/hypo.py", line 30, in test_multilistener_should_not_mutate_nested
    assert len(nested_multi.listeners) == original_length, \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Expected 2 listeners, but found 0
Falsifying example: test_multilistener_should_not_mutate_nested(
    dummy=None,
)
```
</details>

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

<details>

<summary>
Nested MultiListener's listeners list gets cleared from 2 to 0
</summary>
```
Before: nested_multi has 2 listeners
After: nested_multi has 0 listeners
Bug: nested_multi.listeners was cleared!
```
</details>

## Why This Is A Bug

This violates expected Python behavior where constructors should not destructively modify their input arguments. The issue occurs in `MultiListener.__post_init__` at line 116 of `/home/npc/pbt/agentic-pbt/envs/anyio_env/lib/python3.13/site-packages/anyio/streams/stapled.py`:

```python
del listener.listeners[:]  # type: ignore[attr-defined]
```

While the class documentation states "Any MultiListeners in the given collection of listeners will have their listeners moved into this one", the word "moved" does not clearly indicate destructive mutation. In Python, constructors typically do not mutate their arguments, and users would reasonably expect to be able to:

1. Reuse a MultiListener object after passing it to another MultiListener
2. Pass the same MultiListener to multiple other MultiListeners
3. Keep references to nested MultiListeners for monitoring or management purposes

The mutation also creates potential thread safety issues if the nested MultiListener is being used concurrently elsewhere in the application.

## Relevant Context

The MultiListener class is designed to combine multiple listeners into a single listener that serves connections from all of them. The flattening behavior (extracting listeners from nested MultiListeners) is useful, but the destructive mutation is unexpected.

The code at lines 112-120 shows the problematic logic:
- Line 115: `listeners.extend(listener.listeners)` - copies the listener references
- Line 116: `del listener.listeners[:]` - clears the original list (the bug)
- Line 120: Assigns the flattened list to self.listeners

The MultiListener documentation is available at: https://anyio.readthedocs.io/en/stable/api.html#anyio.streams.stapled.MultiListener

## Proposed Fix

Remove the destructive mutation while preserving the flattening behavior:

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