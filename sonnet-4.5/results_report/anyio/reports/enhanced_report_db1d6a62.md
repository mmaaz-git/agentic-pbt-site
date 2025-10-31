# Bug Report: anyio.streams.stapled.MultiListener Mutates Nested Input Objects

**Target**: `anyio.streams.stapled.MultiListener`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `MultiListener` constructor mutates nested `MultiListener` objects passed as input by clearing their internal `listeners` list, violating the principle that constructors should not modify their arguments.

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


if __name__ == "__main__":
    test_multilistener_does_not_mutate_inputs()
```

<details>

<summary>
**Failing input**: `num_listeners=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/58/hypo.py", line 30, in <module>
    test_multilistener_does_not_mutate_inputs()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/58/hypo.py", line 18, in test_multilistener_does_not_mutate_inputs
    def test_multilistener_does_not_mutate_inputs(num_listeners: int):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/58/hypo.py", line 25, in test_multilistener_does_not_mutate_inputs
    assert len(nested.listeners) == original_count, \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: MultiListener mutated input: expected 1 listeners, but found 0
Falsifying example: test_multilistener_does_not_mutate_inputs(
    num_listeners=1,
)
```
</details>

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


# Create a nested MultiListener with two mock listeners
nested_multi = MultiListener(listeners=[MockListener(), MockListener()])
print(f"Before creating outer MultiListener: {len(nested_multi.listeners)} listeners")

# Create an outer MultiListener that contains the nested one
outer_multi = MultiListener(listeners=[nested_multi, MockListener()])
print(f"After creating outer MultiListener: {len(nested_multi.listeners)} listeners")

# Show the impact
print(f"\nOriginal nested MultiListener now has: {nested_multi.listeners}")
print(f"Outer MultiListener has: {len(outer_multi.listeners)} listeners")
```

<details>

<summary>
Nested MultiListener's listeners list is cleared after being passed to another MultiListener
</summary>
```
Before creating outer MultiListener: 2 listeners
After creating outer MultiListener: 0 listeners

Original nested MultiListener now has: []
Outer MultiListener has: 3 listeners
```
</details>

## Why This Is A Bug

This violates expected behavior because Python constructors conventionally do not mutate their input arguments unless explicitly documented with clear warnings. The `MultiListener.__post_init__` method at line 116 in `/home/npc/miniconda/lib/python3.13/site-packages/anyio/streams/stapled.py` contains `del listener.listeners[:]` which destructively clears the nested MultiListener's internal list.

While the docstring mentions that nested listeners will be "moved" (lines 99-100), this terminology is ambiguous and doesn't clearly indicate that the source object will be rendered unusable. In Python, "move" semantics typically refer to logical transfer rather than destructive modification. Standard library methods like `list.extend()` don't clear their source, and users reasonably expect similar behavior here.

This mutation makes nested MultiListeners unusable after being passed to a parent MultiListener, breaking any code that retains references to them. The behavior violates the principle of least surprise and can lead to subtle bugs in user code where MultiListeners unexpectedly become empty.

## Relevant Context

The problematic code is in the `__post_init__` method (lines 111-120) of `/home/npc/miniconda/lib/python3.13/site-packages/anyio/streams/stapled.py`:

```python
def __post_init__(self) -> None:
    listeners: list[Listener[T_Stream]] = []
    for listener in self.listeners:
        if isinstance(listener, MultiListener):
            listeners.extend(listener.listeners)  # Line 115: Copies listeners
            del listener.listeners[:]  # Line 116: MUTATES input by clearing it
        else:
            listeners.append(listener)
    self.listeners = listeners
```

The intent appears to be flattening nested MultiListeners, which is achieved at line 115. However, line 116 goes further by clearing the source, presumably to prevent the nested MultiListener from being used independently. This is overly aggressive and unexpected.

Documentation: https://anyio.readthedocs.io/en/stable/api.html#anyio.streams.stapled.MultiListener

## Proposed Fix

Remove the mutation of input MultiListener objects while preserving the flattening behavior:

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