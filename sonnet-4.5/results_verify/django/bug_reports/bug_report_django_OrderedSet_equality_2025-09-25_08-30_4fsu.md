# Bug Report: Django OrderedSet Missing Equality Implementation

**Target**: `django.utils.datastructures.OrderedSet`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `OrderedSet` class does not implement `__eq__` and `__hash__`, causing two OrderedSets with identical elements in the same order to be unequal when compared with `==`.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
from django.utils.datastructures import OrderedSet


@given(st.lists(st.integers()))
@settings(max_examples=1000)
def test_orderedset_equality_reflexive(items):
    os1 = OrderedSet(items)
    os2 = OrderedSet(items)
    assert os1 == os2, f"OrderedSet({items}) should equal OrderedSet({items})"
```

**Failing input**: `items=[]` (or any other list)

## Reproducing the Bug

```python
from django.utils.datastructures import OrderedSet

os1 = OrderedSet([1, 2, 3])
os2 = OrderedSet([1, 2, 3])

print(f"os1 == os2: {os1 == os2}")
assert os1 == os2, "Two OrderedSets with identical elements should be equal"
```

Output:
```
os1 == os2: False
AssertionError: Two OrderedSets with identical elements should be equal
```

## Why This Is A Bug

As a set-like data structure, `OrderedSet` should implement proper equality semantics. Users would reasonably expect that two `OrderedSet` instances containing the same elements in the same order would be equal. Without `__eq__`, the class falls back to identity comparison (`is`), making it impossible to meaningfully compare OrderedSets. This violates the fundamental contract of collection types in Python, where equality is based on contents, not identity.

## Fix

```diff
--- a/django/utils/datastructures.py
+++ b/django/utils/datastructures.py
@@ -54,6 +54,15 @@ class OrderedSet:
     def __repr__(self):
         data = repr(list(self.dict)) if self.dict else ""
         return f"{self.__class__.__qualname__}({data})"
+
+    def __eq__(self, other):
+        if isinstance(other, OrderedSet):
+            return list(self.dict) == list(other.dict)
+        return NotImplemented
+
+    def __hash__(self):
+        return hash(tuple(self.dict))
```