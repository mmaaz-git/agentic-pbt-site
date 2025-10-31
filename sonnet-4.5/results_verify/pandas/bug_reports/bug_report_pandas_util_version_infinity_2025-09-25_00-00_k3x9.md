# Bug Report: pandas.util.version Infinity Comparison Reflexivity Violation

**Target**: `pandas.util.version.InfinityType` and `pandas.util.version.NegativeInfinityType`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `Infinity` and `NegativeInfinity` singleton objects violate the reflexivity property of comparison operators. `Infinity > Infinity` returns `True` (should be `False`), and `NegativeInfinity < NegativeInfinity` returns `True` (should be `False`). Similarly, `Infinity <= Infinity` returns `False` (should be `True`) and `NegativeInfinity >= NegativeInfinity` returns `False` (should be `True`).

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas.util.version as version_module

def test_infinity_self_comparison():
    inf = version_module.Infinity

    assert inf == inf
    assert not (inf < inf)
    assert not (inf > inf)
    assert inf <= inf
    assert inf >= inf

def test_negative_infinity_self_comparison():
    ninf = version_module.NegativeInfinity

    assert ninf == ninf
    assert not (ninf < ninf)
    assert not (ninf > ninf)
    assert ninf <= ninf
    assert ninf >= ninf
```

**Failing assertions**:
- `assert not (inf > inf)` - fails because `Infinity > Infinity` is `True`
- `assert inf <= inf` - fails because `Infinity <= Infinity` is `False`
- `assert not (ninf < ninf)` - fails because `NegativeInfinity < NegativeInfinity` is `True`
- `assert ninf >= ninf` - fails because `NegativeInfinity >= NegativeInfinity` is `False`

## Reproducing the Bug

```python
import pandas.util.version as version_module

inf = version_module.Infinity
ninf = version_module.NegativeInfinity

print(f"Infinity > Infinity: {inf > inf}")
print(f"Infinity <= Infinity: {inf <= inf}")
print(f"NegativeInfinity < NegativeInfinity: {ninf < ninf}")
print(f"NegativeInfinity >= NegativeInfinity: {ninf >= ninf}")
```

Output:
```
Infinity > Infinity: True
Infinity <= Infinity: False
NegativeInfinity < NegativeInfinity: True
NegativeInfinity >= NegativeInfinity: False
```

## Why This Is A Bug

For any value `x`, the reflexivity property of comparison operators requires:
- `x < x` must be `False`
- `x > x` must be `False`
- `x <= x` must be `True`
- `x >= x` must be `True`
- `x == x` must be `True`

The current implementation violates these properties for the `Infinity` and `NegativeInfinity` singletons. This could lead to incorrect behavior in sorting algorithms, binary search, and other code that relies on standard comparison semantics.

## Fix

```diff
--- a/pandas/util/version/__init__.py
+++ b/pandas/util/version/__init__.py
@@ -30,10 +30,10 @@ class InfinityType:
         return hash(repr(self))

     def __lt__(self, other: object) -> bool:
         return False

     def __le__(self, other: object) -> bool:
-        return False
+        return isinstance(other, type(self))

     def __eq__(self, other: object) -> bool:
         return isinstance(other, type(self))
@@ -42,10 +42,10 @@ class InfinityType:
         return not isinstance(other, type(self))

     def __gt__(self, other: object) -> bool:
-        return True
+        return not isinstance(other, type(self))

     def __ge__(self, other: object) -> bool:
         return True

     def __neg__(self: object) -> NegativeInfinityType:
         return NegativeInfinity
@@ -58,10 +58,10 @@ class NegativeInfinityType:
         return hash(repr(self))

     def __lt__(self, other: object) -> bool:
-        return True
+        return not isinstance(other, type(self))

     def __le__(self, other: object) -> bool:
         return True
@@ -72,10 +72,10 @@ class NegativeInfinityType:
     def __ne__(self, other: object) -> bool:
         return not isinstance(other, type(self))

     def __gt__(self, other: object) -> bool:
         return False

     def __ge__(self, other: object) -> bool:
-        return False
+        return isinstance(other, type(self))

     def __neg__(self: object) -> InfinityType:
         return Infinity
```