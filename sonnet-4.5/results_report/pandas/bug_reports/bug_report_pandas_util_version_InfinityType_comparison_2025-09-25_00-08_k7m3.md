# Bug Report: pandas.util.version InfinityType Comparison Inconsistency

**Target**: `pandas.util.version.InfinityType` and `pandas.util.version.NegativeInfinityType`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `InfinityType` and `NegativeInfinityType` classes violate fundamental comparison operator consistency: when `a == b` is True, both `a <= b` and `a >= b` must be True. However, `Infinity <= Infinity` returns False despite `Infinity == Infinity` being True, and `NegativeInfinity >= NegativeInfinity` returns False despite `NegativeInfinity == NegativeInfinity` being True.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas.util.version as version_module

@given(st.sampled_from([
    version_module.Infinity,
    version_module.NegativeInfinity
]))
def test_comparison_reflexivity(x):
    if x == x:
        assert x <= x, f"{x} should be <= itself when it equals itself"
        assert x >= x, f"{x} should be >= itself when it equals itself"
```

**Failing inputs**: `Infinity` and `NegativeInfinity`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/path/to/pandas')
import pandas.util.version as v

inf = v.Infinity
print(f"Infinity == Infinity: {inf == inf}")
print(f"Infinity <= Infinity: {inf <= inf}")

neginf = v.NegativeInfinity
print(f"NegativeInfinity == NegativeInfinity: {neginf == neginf}")
print(f"NegativeInfinity >= NegativeInfinity: {neginf >= neginf}")
```

**Output:**
```
Infinity == Infinity: True
Infinity <= Infinity: False
NegativeInfinity == NegativeInfinity: True
NegativeInfinity >= NegativeInfinity: False
```

## Why This Is A Bug

The mathematical property of comparison operators requires that for any values `a` and `b`, if `a == b`, then both `a <= b` and `a >= b` must be True. This is a fundamental axiom of total ordering. The current implementation violates this property, which can lead to unexpected behavior when these objects are used in sorting, comparisons, or any algorithm that relies on consistent comparison semantics.

## Fix

```diff
--- a/pandas/util/version/__init__.py
+++ b/pandas/util/version/__init__.py
@@ -32,10 +32,10 @@ class InfinityType:
     def __lt__(self, other: object) -> bool:
         return False

     def __le__(self, other: object) -> bool:
-        return False
+        return isinstance(other, type(self))

     def __eq__(self, other: object) -> bool:
         return isinstance(other, type(self))

@@ -76,10 +76,10 @@ class NegativeInfinityType:
     def __gt__(self, other: object) -> bool:
         return False

     def __ge__(self, other: object) -> bool:
-        return False
+        return isinstance(other, type(self))

     def __neg__(self: object) -> InfinityType:
         return Infinity
```