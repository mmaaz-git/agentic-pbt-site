# Bug Report: dask.callbacks.Callback Multiple Instances Unregister

**Target**: `dask.callbacks.Callback`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

When multiple `Callback` instances with identical callback functions are registered, calling `unregister()` on one instance causes subsequent `unregister()` calls on other instances to raise a `KeyError`.

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
```

**Failing input**: `n=2`

## Reproducing the Bug

```python
from dask.callbacks import Callback

cb1 = Callback()
cb2 = Callback()

cb1.register()
cb2.register()

cb1.unregister()
cb2.unregister()
```

Output:
```
KeyError: (None, None, None, None, None)
```

## Why This Is A Bug

Multiple `Callback` instances created with the same parameters (all None) share an identical `_callback` tuple value of `(None, None, None, None, None)`. Since `Callback.active` is a set and uses `_callback` as the identity, registering two such callbacks only adds one entry to the set. When the first callback is unregistered, it removes the shared tuple from the set. The second callback's `unregister()` then fails with a `KeyError` because the tuple is no longer present.

This violates the reasonable expectation that each callback instance should be independently registrable and unregistrable without affecting other instances.

## Fix

The `unregister()` method should handle the case where the callback might have already been removed. A simple fix is to use `discard()` instead of `remove()`:

```diff
--- a/dask/callbacks.py
+++ b/dask/callbacks.py
@@ -80,7 +80,7 @@ class Callback:
         Callback.active.add(self._callback)

     def unregister(self) -> None:
-        Callback.active.remove(self._callback)
+        Callback.active.discard(self._callback)
```

This change makes `unregister()` idempotent and prevents the `KeyError` when multiple callbacks share the same `_callback` tuple.