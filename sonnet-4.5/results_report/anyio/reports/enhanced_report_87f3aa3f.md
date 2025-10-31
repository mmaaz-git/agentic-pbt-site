# Bug Report: anyio.streams.stapled.MultiListener Mutates Input MultiListener Objects

**Target**: `anyio.streams.stapled.MultiListener`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `MultiListener` constructor unexpectedly mutates any `MultiListener` objects passed as input by emptying their `listeners` list, violating the Python convention that constructors should not modify their input arguments.

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

    # This assertion should pass but fails due to the bug
    assert len(multi1.listeners) == original_count, f"Expected {original_count} listeners in multi1, but found {len(multi1.listeners)}"


if __name__ == "__main__":
    test_multilistener_does_not_mutate_input()
```

<details>

<summary>
**Failing input**: `num_listeners=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/6/hypo.py", line 36, in <module>
    test_multilistener_does_not_mutate_input()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/6/hypo.py", line 22, in test_multilistener_does_not_mutate_input
    def test_multilistener_does_not_mutate_input(num_listeners):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/6/hypo.py", line 32, in test_multilistener_does_not_mutate_input
    assert len(multi1.listeners) == original_count, f"Expected {original_count} listeners in multi1, but found {len(multi1.listeners)}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Expected 1 listeners in multi1, but found 0
Falsifying example: test_multilistener_does_not_mutate_input(
    num_listeners=1,
)
```
</details>

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

# Create multi1 with two listeners
multi1 = MultiListener(listeners=[listener1, listener2])
print(f"multi1.listeners before creating multi2: {multi1.listeners}")
print(f"multi1.listeners length before: {len(multi1.listeners)}")

# Create multi2 with multi1 as input
multi2 = MultiListener(listeners=[multi1])
print(f"multi1.listeners after creating multi2: {multi1.listeners}")
print(f"multi1.listeners length after: {len(multi1.listeners)}")
print(f"multi2.listeners: {multi2.listeners}")
print(f"multi2.listeners length: {len(multi2.listeners)}")

# This assertion demonstrates the bug
assert len(multi1.listeners) == 0, f"multi1.listeners was mutated to empty! Length is {len(multi1.listeners)}"
print("\nBUG CONFIRMED: multi1.listeners was emptied when passed to multi2!")
```

<details>

<summary>
Output shows multi1.listeners is emptied after being passed to multi2
</summary>
```
multi1.listeners before creating multi2: [MockListener(name='listener1'), MockListener(name='listener2')]
multi1.listeners length before: 2
multi1.listeners after creating multi2: []
multi1.listeners length after: 0
multi2.listeners: [MockListener(name='listener1'), MockListener(name='listener2')]
multi2.listeners length: 2

BUG CONFIRMED: multi1.listeners was emptied when passed to multi2!
```
</details>

## Why This Is A Bug

This behavior violates several important principles and expectations:

1. **Python Convention Violation**: In Python, constructors should not mutate their input arguments unless this is explicitly and clearly documented. The `MultiListener.__post_init__` method modifies nested `MultiListener` objects passed as input by calling `del listener.listeners[:]` on line 116 of `/home/npc/pbt/agentic-pbt/envs/anyio_env/stapled.py`.

2. **Documentation Ambiguity**: While the docstring mentions that nested MultiListeners will have their listeners "moved into" the new MultiListener, this doesn't clearly indicate destructive mutation. The word "moved" could reasonably be interpreted as copying/flattening rather than destroying the original.

3. **Type Hint Mismatch**: The parameter is typed as `Sequence[Listener[T_Stream]]`, and sequences are generally expected to be read-only in constructor contexts. Mutating elements within a sequence parameter is unexpected.

4. **Object Integrity**: After passing a `MultiListener` to create another `MultiListener`, the original becomes an empty shell with no listeners, making it unusable for any further operations.

5. **Composability Issues**: This prevents safely composing or reusing `MultiListener` objects, as they become invalid after being used as input to another `MultiListener`.

## Relevant Context

The bug occurs in the `MultiListener.__post_init__` method at line 116 of `anyio/streams/stapled.py`:

```python
def __post_init__(self) -> None:
    listeners: list[Listener[T_Stream]] = []
    for listener in self.listeners:
        if isinstance(listener, MultiListener):
            listeners.extend(listener.listeners)
            del listener.listeners[:]  # Line 116 - THE PROBLEMATIC LINE
        else:
            listeners.append(listener)
    self.listeners = listeners
```

The flattening behavior (combining nested MultiListeners) is desirable and should be preserved. The issue is solely with the mutation of the input object via `del listener.listeners[:]`.

Similar patterns in Python standard library (like `itertools.chain` or `list.extend`) don't mutate their inputs, setting the expectation that this shouldn't either.

## Proposed Fix

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