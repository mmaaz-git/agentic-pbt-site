# Bug Report: pandas.util.version InfinityType Comparison Inconsistency

**Target**: `pandas.util.version.InfinityType` and `pandas.util.version.NegativeInfinityType`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `InfinityType` and `NegativeInfinityType` classes violate Python's comparison semantics: when two instances are equal (`a == b`), the comparison operators `<`, `>`, `<=`, `>=` produce inconsistent results. Specifically, `InfinityType()` instances always return `True` for `>` comparisons and `NegativeInfinityType()` instances always return `True` for `<` comparisons, even when comparing equal instances.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pandas.util.version import InfinityType, NegativeInfinityType


@given(st.integers(min_value=0, max_value=100))
@settings(max_examples=200)
def test_infinity_comparison_consistency(n):
    inf1 = InfinityType()
    inf2 = InfinityType()

    if inf1 == inf2:
        assert not (inf1 > inf2)
        assert not (inf1 < inf2)
        assert inf1 <= inf2
        assert inf1 >= inf2


@given(st.integers(min_value=0, max_value=100))
@settings(max_examples=200)
def test_negative_infinity_comparison_consistency(n):
    ninf1 = NegativeInfinityType()
    ninf2 = NegativeInfinityType()

    if ninf1 == ninf2:
        assert not (ninf1 > ninf2)
        assert not (ninf1 < ninf2)
        assert ninf1 <= ninf2
        assert ninf1 >= ninf2
```

**Failing input**: Any pair of `InfinityType()` or `NegativeInfinityType()` instances

## Reproducing the Bug

```python
from pandas.util.version import InfinityType, NegativeInfinityType

inf1 = InfinityType()
inf2 = InfinityType()

print(f"inf1 == inf2: {inf1 == inf2}")
print(f"inf1 > inf2: {inf1 > inf2}")
print(f"inf1 < inf2: {inf1 < inf2}")
print(f"inf1 <= inf2: {inf1 <= inf2}")
print(f"inf1 >= inf2: {inf1 >= inf2}")

assert inf1 == inf2
assert inf1 > inf2

ninf1 = NegativeInfinityType()
ninf2 = NegativeInfinityType()

print(f"\nninf1 == ninf2: {ninf1 == ninf2}")
print(f"ninf1 < ninf2: {ninf1 < ninf2}")
print(f"ninf1 <= ninf2: {ninf1 <= ninf2}")

assert ninf1 == ninf2
assert ninf1 < ninf2
```

## Why This Is A Bug

Python's comparison semantics require that if `a == b`, then:
- `a > b` must be `False`
- `a < b` must be `False`
- `a >= b` must be `True`
- `a <= b` must be `True`

The current implementation violates this because the comparison methods (`__lt__`, `__gt__`, etc.) return constant boolean values without checking if the other object is also an instance of the same infinity type. This creates logically inconsistent comparison results.

## Fix

```diff
--- a/pandas/util/version/__init__.py
+++ b/pandas/util/version/__init__.py
@@ -27,20 +27,26 @@ class InfinityType:
         return hash(repr(self))

     def __lt__(self, other: object) -> bool:
+        if isinstance(other, InfinityType):
+            return False
         return False

     def __le__(self, other: object) -> bool:
+        if isinstance(other, InfinityType):
+            return True
         return False

     def __eq__(self, other: object) -> bool:
         return isinstance(other, type(self))

     def __ne__(self, other: object) -> bool:
         return not isinstance(other, type(self))

     def __gt__(self, other: object) -> bool:
+        if isinstance(other, InfinityType):
+            return False
         return True

     def __ge__(self, other: object) -> bool:
+        if isinstance(other, InfinityType):
+            return True
         return True

     def __neg__(self: object) -> NegativeInfinityType:
@@ -55,20 +61,26 @@ class NegativeInfinityType:
         return hash(repr(self))

     def __lt__(self, other: object) -> bool:
+        if isinstance(other, NegativeInfinityType):
+            return False
         return True

     def __le__(self, other: object) -> bool:
+        if isinstance(other, NegativeInfinityType):
+            return True
         return True

     def __eq__(self, other: object) -> bool:
         return isinstance(other, type(self))

     def __ne__(self, other: object) -> bool:
         return not isinstance(other, type(self))

     def __gt__(self, other: object) -> bool:
+        if isinstance(other, NegativeInfinityType):
+            return False
         return False

     def __ge__(self, other: object) -> bool:
+        if isinstance(other, NegativeInfinityType):
+            return True
         return False

     def __neg__(self: object) -> InfinityType:
```