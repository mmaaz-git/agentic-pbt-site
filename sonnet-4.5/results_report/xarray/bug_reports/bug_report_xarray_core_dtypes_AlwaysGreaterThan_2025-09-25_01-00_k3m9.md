# Bug Report: xarray.core.dtypes AlwaysGreaterThan/AlwaysLessThan Ordering Inconsistency

**Target**: `xarray.core.dtypes.AlwaysGreaterThan` and `xarray.core.dtypes.AlwaysLessThan`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `AlwaysGreaterThan` and `AlwaysLessThan` classes violate fundamental ordering properties: when two instances are equal (according to `__eq__`), they should not satisfy `>` or `<` comparisons. However, `AlwaysGreaterThan.__gt__` returns `True` even when comparing equal instances, and `AlwaysLessThan.__lt__` does the same.

## Property-Based Test

```python
def test_always_greater_than_transitivity():
    inf1 = dtypes.AlwaysGreaterThan()
    inf2 = dtypes.AlwaysGreaterThan()
    assert inf1 == inf2
    assert not (inf1 > inf2)  # FAILS - should be False but is True
    assert not (inf1 < inf2)


def test_always_less_than_transitivity():
    ninf1 = dtypes.AlwaysLessThan()
    ninf2 = dtypes.AlwaysLessThan()
    assert ninf1 == ninf2
    assert not (ninf1 < ninf2)  # FAILS - should be False but is True
    assert not (ninf1 > ninf2)
```

**Failing input**: Two instances of the same class

## Reproducing the Bug

```python
from xarray.core import dtypes

inf1 = dtypes.AlwaysGreaterThan()
inf2 = dtypes.AlwaysGreaterThan()

print(inf1 == inf2)  # True
print(inf1 > inf2)   # True - BUG: should be False

ninf1 = dtypes.AlwaysLessThan()
ninf2 = dtypes.AlwaysLessThan()

print(ninf1 == ninf2)  # True
print(ninf1 < ninf2)   # True - BUG: should be False
```

## Why This Is A Bug

This violates the fundamental property of total orderings: if `a == b`, then `a > b` must be `False` (and `a < b` must be `False`). The classes are decorated with `@functools.total_ordering`, which expects this property to hold. The current implementation creates inconsistent comparison behavior that can lead to incorrect sorting and comparison results.

## Fix

The `__gt__` and `__lt__` methods should check for equality before returning `True`:

```diff
--- a/xarray/core/dtypes.py
+++ b/xarray/core/dtypes.py
@@ -17,7 +17,8 @@ from xarray.core import utils
 @functools.total_ordering
 class AlwaysGreaterThan:
     def __gt__(self, other):
-        return True
+        if self == other:
+            return False
+        return True

     def __eq__(self, other):
         return isinstance(other, type(self))
@@ -26,7 +27,8 @@ class AlwaysGreaterThan:
 @functools.total_ordering
 class AlwaysLessThan:
     def __lt__(self, other):
-        return True
+        if self == other:
+            return False
+        return True

     def __eq__(self, other):
         return isinstance(other, type(self))
```