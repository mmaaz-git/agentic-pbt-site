# Bug Report: xarray.core.dtypes AlwaysGreaterThan/AlwaysLessThan Total Ordering Violation

**Target**: `xarray.core.dtypes.AlwaysGreaterThan` and `xarray.core.dtypes.AlwaysLessThan`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `AlwaysGreaterThan` and `AlwaysLessThan` classes violate the antisymmetry property of total ordering despite being decorated with `@functools.total_ordering`. When two instances are equal according to `__eq__`, they incorrectly report being greater than (or less than) each other.

## Property-Based Test

```python
from hypothesis import given
from xarray.core.dtypes import AlwaysGreaterThan, AlwaysLessThan

def test_always_greater_than_equals_itself():
    inf1 = AlwaysGreaterThan()
    inf2 = AlwaysGreaterThan()
    assert inf1 == inf2
    assert not (inf1 != inf2)
    assert not (inf1 < inf2)
    assert not (inf1 > inf2)  # FAILS: inf1 > inf2 is True

def test_always_less_than_equals_itself():
    ninf1 = AlwaysLessThan()
    ninf2 = AlwaysLessThan()
    assert ninf1 == ninf2
    assert not (ninf1 != ninf2)
    assert not (ninf1 < ninf2)  # FAILS: ninf1 < ninf2 is True
    assert not (ninf1 > ninf2)
```

**Failing input**: Any two instances of the same class

## Reproducing the Bug

```python
from xarray.core.dtypes import AlwaysGreaterThan, AlwaysLessThan

inf1 = AlwaysGreaterThan()
inf2 = AlwaysGreaterThan()

print(f"inf1 == inf2: {inf1 == inf2}")  # True
print(f"inf1 > inf2: {inf1 > inf2}")    # True (BUG!)
print(f"inf1 <= inf2: {inf1 <= inf2}")  # False (BUG!)

ninf1 = AlwaysLessThan()
ninf2 = AlwaysLessThan()

print(f"ninf1 == ninf2: {ninf1 == ninf2}")  # True
print(f"ninf1 < ninf2: {ninf1 < ninf2}")    # True (BUG!)
print(f"ninf1 >= ninf2: {ninf1 >= ninf2}")  # False (BUG!)
```

## Why This Is A Bug

The `@functools.total_ordering` decorator requires that comparison methods satisfy the antisymmetry property: if `a == b`, then `a > b` must be `False` and `a >= b` must be `True`.

Currently:
- `AlwaysGreaterThan.__gt__(other)` always returns `True`, even when `other` is also an `AlwaysGreaterThan` instance
- `AlwaysLessThan.__lt__(other)` always returns `True`, even when `other` is also an `AlwaysLessThan` instance

This violates the contract of total ordering and can lead to inconsistent behavior in sorting, comparison chains, and other operations that rely on total ordering semantics.

## Fix

```diff
@functools.total_ordering
class AlwaysGreaterThan:
    def __gt__(self, other):
+       if isinstance(other, type(self)):
+           return False
        return True

    def __eq__(self, other):
        return isinstance(other, type(self))


@functools.total_ordering
class AlwaysLessThan:
    def __lt__(self, other):
+       if isinstance(other, type(self)):
+           return False
        return True

    def __eq__(self, other):
        return isinstance(other, type(self))
```