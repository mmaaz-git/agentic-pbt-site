# Bug Report: dask.callbacks.Callback Multiple Instance Unregister KeyError

**Target**: `dask.callbacks.Callback.unregister()`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

When multiple `Callback` instances with identical callback functions are created and registered, calling `unregister()` on one instance causes subsequent `unregister()` calls on other instances to raise a `KeyError`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from dask.callbacks import Callback


@given(st.integers(min_value=1, max_value=10))
def test_multiple_callbacks_register_unregister(n):
    callbacks = [Callback() for _ in range(n)]
    initial_active = Callback.active.copy()

    for cb in callbacks:
        cb.register()

    for cb in callbacks:
        assert cb._callback in Callback.active

    for cb in callbacks:
        cb.unregister()

    assert Callback.active == initial_active


if __name__ == "__main__":
    # Run the test
    test_multiple_callbacks_register_unregister()
```

<details>

<summary>
**Failing input**: `n=2`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 24, in <module>
    test_multiple_callbacks_register_unregister()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 6, in test_multiple_callbacks_register_unregister
    def test_multiple_callbacks_register_unregister(n):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 17, in test_multiple_callbacks_register_unregister
    cb.unregister()
    ~~~~~~~~~~~~~^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/callbacks.py", line 83, in unregister
    Callback.active.remove(self._callback)
    ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^
KeyError: (None, None, None, None, None)
Falsifying example: test_multiple_callbacks_register_unregister(
    n=2,
)
```
</details>

## Reproducing the Bug

```python
from dask.callbacks import Callback

# Create two identical Callback instances
cb1 = Callback()
cb2 = Callback()

# Print their _callback tuples to show they are identical
print(f"cb1._callback: {cb1._callback}")
print(f"cb2._callback: {cb2._callback}")
print(f"Are they equal? {cb1._callback == cb2._callback}")
print()

# Register both callbacks
print("Registering cb1...")
cb1.register()
print(f"Callback.active after cb1.register(): {Callback.active}")

print("\nRegistering cb2...")
cb2.register()
print(f"Callback.active after cb2.register(): {Callback.active}")
print(f"Number of entries in active set: {len(Callback.active)}")

# Unregister the first callback
print("\nUnregistering cb1...")
cb1.unregister()
print(f"Callback.active after cb1.unregister(): {Callback.active}")

# Try to unregister the second callback - this will raise KeyError
print("\nUnregistering cb2...")
try:
    cb2.unregister()
    print("cb2 unregistered successfully")
except KeyError as e:
    print(f"KeyError raised: {e}")
```

<details>

<summary>
KeyError raised when unregistering second callback
</summary>
```
cb1._callback: (None, None, None, None, None)
cb2._callback: (None, None, None, None, None)
Are they equal? True

Registering cb1...
Callback.active after cb1.register(): {(None, None, None, None, None)}

Registering cb2...
Callback.active after cb2.register(): {(None, None, None, None, None)}
Number of entries in active set: 1

Unregistering cb1...
Callback.active after cb1.unregister(): set()

Unregistering cb2...
KeyError raised: (None, None, None, None, None)
```
</details>

## Why This Is A Bug

This behavior violates the expected contract of the Callback API in several ways:

1. **Object Independence Violation**: The documentation shows examples of using `cb.register()` followed by `cb.unregister()` (line 37-38 of callbacks.py), implying each callback instance should be independently manageable. Users reasonably expect that each `Callback` instance operates independently, but the current implementation breaks this expectation.

2. **Silent Identity Collision**: Multiple `Callback()` instances created with default parameters (all `None`) produce identical `_callback` tuples of `(None, None, None, None, None)`. Since `Callback.active` is a set that uses these tuples as identity, registering multiple callbacks only adds one entry. This identity collision is not documented or warned about.

3. **Asymmetric Register/Unregister**: While `register()` silently handles duplicate registrations (set.add() is idempotent), `unregister()` crashes on missing entries. This asymmetry creates a trap for users.

4. **Internal Inconsistency**: The codebase itself recognizes this issue - the `add_callbacks` context manager uses `discard()` instead of `remove()` on line 145 to avoid this exact problem, showing the safer pattern is already known and used elsewhere.

## Relevant Context

- **Source Code Location**: `/dask/callbacks.py`, line 83
- **Affected Method**: `Callback.unregister()`
- **Related Code**: The `add_callbacks` class (line 120-146) already handles this correctly using `discard()` in its `__exit__` method (line 145)
- **Common Scenarios**: This bug commonly affects:
  - Unit tests that create multiple callback instances
  - Dynamic callback creation in loops or parallel processing
  - Libraries that create callbacks with default parameters
- **Documentation**: The class docstring (lines 11-48) doesn't warn about or document this limitation

## Proposed Fix

```diff
--- a/dask/callbacks.py
+++ b/dask/callbacks.py
@@ -80,7 +80,7 @@ class Callback:
         Callback.active.add(self._callback)

     def unregister(self) -> None:
-        Callback.active.remove(self._callback)
+        Callback.active.discard(self._callback)


 def unpack_callbacks(cbs):
```