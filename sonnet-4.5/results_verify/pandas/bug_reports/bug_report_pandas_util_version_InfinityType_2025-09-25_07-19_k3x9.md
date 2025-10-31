# Bug Report: pandas.util.version.InfinityType Violates Trichotomy Law

**Target**: `pandas.util.version.InfinityType`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`InfinityType.__gt__` always returns `True`, even when comparing Infinity with itself, violating the trichotomy law that requires exactly one of `<`, `==`, or `>` to be true.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas.util.version as pv

def test_infinity_equals_itself():
    inf = pv.Infinity
    assert inf == inf
    assert not (inf != inf)
    assert not (inf < inf)
    assert not (inf > inf)
    assert inf <= inf
    assert inf >= inf
```

**Failing input**: `pv.Infinity` compared with itself

## Reproducing the Bug

```python
import pandas.util.version as pv

inf = pv.Infinity

print(f"inf == inf: {inf == inf}")
print(f"inf > inf: {inf > inf}")

assert inf == inf
assert not (inf > inf)
```

## Why This Is A Bug

The trichotomy law states that for any two values `a` and `b`, exactly one of `a < b`, `a == b`, or `a > b` must be true. When `a == b` is `True`, both `a < b` and `a > b` must be `False`.

However, `Infinity == Infinity` returns `True` while `Infinity > Infinity` also returns `True`, violating this fundamental property of ordered types.

## Fix

```diff
--- a/pandas/util/version/__init__.py
+++ b/pandas/util/version/__init__.py
@@ -36,7 +36,7 @@ class InfinityType:
         return False

     def __gt__(self, other: object) -> bool:
-        return True
+        return not isinstance(other, type(self))

     def __ge__(self, other: object) -> bool:
         return True
```