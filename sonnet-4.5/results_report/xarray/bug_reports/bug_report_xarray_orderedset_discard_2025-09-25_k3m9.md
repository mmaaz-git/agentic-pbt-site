# Bug Report: xarray.core.utils.OrderedSet.discard() Raises KeyError

**Target**: `xarray.core.utils.OrderedSet.discard()`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `OrderedSet.discard()` method raises `KeyError` when attempting to discard a non-existent element, violating the Python `MutableSet` API contract which requires `discard()` to silently do nothing when the element is not present.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from xarray.core.utils import OrderedSet

@given(st.lists(st.integers()), st.integers())
def test_orderedset_discard_never_raises(initial_values, value_to_discard):
    """
    Property: discard() should never raise an error, whether the element
    exists or not. This is the core contract of MutableSet.discard().
    """
    os = OrderedSet(initial_values)
    os.discard(value_to_discard)
```

**Failing input**: `OrderedSet([1, 2, 3])`, attempting to `discard(999)` where `999` is not in the set.

## Reproducing the Bug

```python
from xarray.core.utils import OrderedSet

os = OrderedSet([1, 2, 3])

os.discard(999)
```

This raises `KeyError: 999`, but according to the `MutableSet` API, it should silently do nothing.

For comparison, the built-in `set` behaves correctly:

```python
s = {1, 2, 3}
s.discard(999)
```

This does not raise any error.

## Why This Is A Bug

According to Python's `collections.abc.MutableSet` documentation, the `discard()` method should "Remove element elem from the set **if it is present**". The emphasis on "if it is present" means the method must not raise an error when the element is absent. This distinguishes `discard()` from `remove()`, which is required to raise `KeyError` for missing elements.

The current implementation at `xarray/core/utils.py:599-600` is:

```python
def discard(self, value: T) -> None:
    del self._d[value]
```

This unconditionally deletes from the dictionary, which raises `KeyError` if `value` is not a key.

## Fix

```diff
--- a/xarray/core/utils.py
+++ b/xarray/core/utils.py
@@ -597,7 +597,7 @@ class OrderedSet(MutableSet[T]):
         self._d[value] = None

     def discard(self, value: T) -> None:
-        del self._d[value]
+        self._d.pop(value, None)

     # Additional methods
```