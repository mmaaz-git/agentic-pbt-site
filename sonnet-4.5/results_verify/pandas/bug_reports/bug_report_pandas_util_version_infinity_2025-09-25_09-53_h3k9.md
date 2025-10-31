# Bug Report: pandas.util.version InfinityType Self-Comparison

**Target**: `pandas.util.version.InfinityType` and `pandas.util.version.NegativeInfinityType`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

InfinityType and NegativeInfinityType violate the mathematical property of total ordering by returning True when comparing instances to themselves with `>` and `<` operators respectively.

## Property-Based Test

```python
import pandas.util.version as version_module

def test_infinity_self_comparison():
    inf = version_module.Infinity

    assert not (inf < inf)
    assert not (inf > inf)
    assert inf == inf
    assert inf <= inf
    assert inf >= inf


def test_negative_infinity_self_comparison():
    neg_inf = version_module.NegativeInfinity

    assert not (neg_inf < neg_inf)
    assert not (neg_inf > neg_inf)
    assert neg_inf == neg_inf
    assert neg_inf <= neg_inf
    assert neg_inf >= neg_inf
```

**Failing assertions**:
- `Infinity > Infinity` returns `True` (expected `False`)
- `NegativeInfinity < NegativeInfinity` returns `True` (expected `False`)

## Reproducing the Bug

```python
import pandas.util.version as version_module

inf = version_module.Infinity
neg_inf = version_module.NegativeInfinity

print(f"Infinity > Infinity: {inf > inf}")
print(f"NegativeInfinity < NegativeInfinity: {neg_inf < neg_inf}")
```

## Why This Is A Bug

In any total ordering, the relation `x < x` must be false for all values x (irreflexivity of strict ordering). Similarly, `x > x` must also be false. The current implementation violates this fundamental property:

- `InfinityType.__gt__` returns `True` for all comparisons, including `Infinity > Infinity`
- `NegativeInfinityType.__lt__` returns `True` for all comparisons, including `NegativeInfinity < NegativeInfinity`

This makes the ordering relation inconsistent and could lead to incorrect sorting or comparison results when these sentinel values are used in version comparison keys.

## Fix

```diff
--- a/pandas/util/version/__init__.py
+++ b/pandas/util/version/__init__.py
@@ -41,10 +41,10 @@ class InfinityType:
     def __ne__(self, other: object) -> bool:
         return not isinstance(other, type(self))

     def __gt__(self, other: object) -> bool:
-        return True
+        return not isinstance(other, type(self))

     def __ge__(self, other: object) -> bool:
         return True

     def __neg__(self: object) -> NegativeInfinityType:
@@ -61,10 +61,10 @@ class NegativeInfinityType:
         return hash(repr(self))

     def __lt__(self, other: object) -> bool:
-        return True
+        return not isinstance(other, type(self))

     def __le__(self, other: object) -> bool:
         return True

     def __gt__(self, other: object) -> bool:
```