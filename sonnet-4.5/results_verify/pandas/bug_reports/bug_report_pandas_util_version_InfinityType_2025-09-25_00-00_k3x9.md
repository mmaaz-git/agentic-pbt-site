# Bug Report: pandas.util.version InfinityType Comparison Operators

**Target**: `pandas.util.version.InfinityType` and `pandas.util.version.NegativeInfinityType`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The InfinityType and NegativeInfinityType classes violate fundamental comparison operator invariants. Specifically, `Infinity <= Infinity` returns False (should be True), `Infinity > Infinity` returns True (should be False), `NegativeInfinity >= NegativeInfinity` returns False (should be True), and `NegativeInfinity < NegativeInfinity` returns True (should be False).

## Property-Based Test

```python
from pandas.util.version import Infinity, NegativeInfinity
from hypothesis import given, strategies as st


@given(st.just(Infinity))
def test_infinity_reflexive_comparisons(inf):
    assert inf == inf
    assert inf <= inf
    assert inf >= inf
    assert not (inf < inf)
    assert not (inf > inf)


@given(st.just(NegativeInfinity))
def test_negative_infinity_reflexive_comparisons(ninf):
    assert ninf == ninf
    assert ninf <= ninf
    assert ninf >= ninf
    assert not (ninf < ninf)
    assert not (ninf > ninf)
```

**Failing inputs**: `Infinity` and `NegativeInfinity` (the singleton instances)

## Reproducing the Bug

```python
from pandas.util.version import Infinity, NegativeInfinity

assert Infinity == Infinity
assert Infinity <= Infinity

assert NegativeInfinity == NegativeInfinity
assert NegativeInfinity >= NegativeInfinity

assert not (Infinity > Infinity)
assert not (NegativeInfinity < NegativeInfinity)
```

## Why This Is A Bug

For any mathematical ordering, reflexive comparisons must satisfy:
- For all x: `x == x` implies `x <= x` and `x >= x`
- For all x: `not (x < x)` and `not (x > x)`

These are fundamental properties of total orderings that Python's comparison operators are expected to satisfy. The current implementation violates these invariants when comparing Infinity/NegativeInfinity with themselves. This could lead to incorrect sorting behavior or set/dict inconsistencies when these sentinel values are used in version comparisons.

## Fix

```diff
--- a/pandas/util/version/__init__.py
+++ b/pandas/util/version/__init__.py
@@ -32,10 +32,12 @@ class InfinityType:
     def __lt__(self, other: object) -> bool:
         return False

     def __le__(self, other: object) -> bool:
-        return False
+        return isinstance(other, type(self))

     def __eq__(self, other: object) -> bool:
         return isinstance(other, type(self))

     def __ne__(self, other: object) -> bool:
         return not isinstance(other, type(self))

     def __gt__(self, other: object) -> bool:
-        return True
+        return not isinstance(other, type(self))

     def __ge__(self, other: object) -> bool:
         return True
@@ -64,10 +66,10 @@ class NegativeInfinityType:
     def __lt__(self, other: object) -> bool:
-        return True
+        return not isinstance(other, type(self))

     def __le__(self, other: object) -> bool:
         return True

     def __eq__(self, other: object) -> bool:
         return isinstance(other, type(self))

     def __ne__(self, other: object) -> bool:
         return not isinstance(other, type(self))

     def __gt__(self, other: object) -> bool:
         return False

     def __ge__(self, other: object) -> bool:
-        return False
+        return isinstance(other, type(self))
```