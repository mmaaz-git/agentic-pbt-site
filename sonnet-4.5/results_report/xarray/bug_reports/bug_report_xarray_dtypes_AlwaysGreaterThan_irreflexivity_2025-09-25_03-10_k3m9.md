# Bug Report: xarray.core.dtypes AlwaysGreaterThan/AlwaysLessThan Comparison Operators Violate Irreflexivity

**Target**: `xarray.core.dtypes.AlwaysGreaterThan` and `xarray.core.dtypes.AlwaysLessThan`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `AlwaysGreaterThan` and `AlwaysLessThan` classes violate the irreflexivity property of strict comparison operators. Specifically, `agt > agt` returns `True` (should be `False`) and `alt < alt` returns `True` (should be `False`). This also causes a violation of the trichotomy law, where both `a == b` and `a > b` can be true simultaneously.

## Property-Based Test

```python
from xarray.core.dtypes import AlwaysGreaterThan, AlwaysLessThan
from hypothesis import given, strategies as st, settings


@settings(max_examples=100)
@given(st.builds(AlwaysGreaterThan))
def test_always_greater_than_irreflexivity(agt):
    """Test that AlwaysGreaterThan satisfies irreflexivity: agt > agt should be False"""
    assert not (agt > agt), "AlwaysGreaterThan violates irreflexivity: agt > agt is True"


@settings(max_examples=100)
@given(st.builds(AlwaysGreaterThan), st.builds(AlwaysGreaterThan))
def test_trichotomy_law(agt1, agt2):
    """Test trichotomy: exactly one of a < b, a == b, a > b should be true"""
    less = agt1 < agt2
    equal = agt1 == agt2
    greater = agt1 > agt2

    true_count = sum([less, equal, greater])
    assert true_count == 1, f"Trichotomy violated: {true_count} conditions are true"
```

**Failing input**: Any `AlwaysGreaterThan()` or `AlwaysLessThan()` instance

## Reproducing the Bug

```python
from xarray.core.dtypes import AlwaysGreaterThan, AlwaysLessThan

agt = AlwaysGreaterThan()
print(f"agt > agt: {agt > agt}")
print(f"agt == agt: {agt == agt}")

alt = AlwaysLessThan()
print(f"alt < alt: {alt < alt}")
print(f"alt == alt: {alt == alt}")

agt1 = AlwaysGreaterThan()
agt2 = AlwaysGreaterThan()
print(f"agt1 > agt2: {agt1 > agt2}")
print(f"agt1 == agt2: {agt1 == agt2}")
print(f"Both are True - trichotomy violated!")
```

Output:
```
agt > agt: True
agt == agt: True
alt < alt: True
alt == alt: True
agt1 > agt2: True
agt1 == agt2: True
Both are True - trichotomy violated!
```

## Why This Is A Bug

This violates fundamental mathematical properties of comparison operators:

1. **Irreflexivity violation**: For any strict ordering (`>`, `<`), the property `a > a` must be False for all `a`. But `AlwaysGreaterThan.__gt__` unconditionally returns True, even when comparing with itself.

2. **Trichotomy violation**: For any total order, exactly one of `a < b`, `a == b`, or `a > b` should be true. But with two `AlwaysGreaterThan` instances, both `a == b` and `a > b` are True.

3. **Decorator contract**: The classes use `@functools.total_ordering` which promises to implement a valid total ordering, but the implementation violates the mathematical requirements.

4. **Potential impact**: These classes are used as sentinel values (`INF` and `NINF`) in xarray's dtype system. Incorrect comparison behavior could lead to:
   - Incorrect sorting when these values are in collections
   - Unexpected behavior in binary search or other algorithms relying on comparison
   - Logical inconsistencies in code that assumes standard comparison semantics

## Fix

The issue is that `__gt__` and `__lt__` return True unconditionally, without checking if `other` is the same instance or an equal object.

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

This ensures that when comparing two `AlwaysGreaterThan` instances (or two `AlwaysLessThan` instances), the comparison returns False and only `__eq__` returns True, satisfying both irreflexivity and trichotomy.