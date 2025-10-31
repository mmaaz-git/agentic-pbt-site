# Bug Report: AlwaysGreaterThan violates total ordering invariants

**Target**: `xarray.core.dtypes.AlwaysGreaterThan`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `AlwaysGreaterThan` class is decorated with `@functools.total_ordering` but violates the fundamental invariant that `a > a` must be `False` for any object `a`. This class returns `True` for `self > self`, breaking the irreflexivity property of strict ordering.

## Property-Based Test

```python
from hypothesis import given
from hypothesis import strategies as st
from xarray.core.dtypes import AlwaysGreaterThan

@given(st.integers())
def test_alwaysgt_comparison_properties(x):
    agt = AlwaysGreaterThan()

    if not isinstance(x, AlwaysGreaterThan):
        assert agt > x

    agt2 = AlwaysGreaterThan()
    assert agt == agt2
    assert not (agt > agt2)
```

**Failing input**: Any two `AlwaysGreaterThan` instances

## Reproducing the Bug

```python
from xarray.core.dtypes import AlwaysGreaterThan

agt1 = AlwaysGreaterThan()
agt2 = AlwaysGreaterThan()

print(f"agt1 == agt2: {agt1 == agt2}")
print(f"agt1 > agt2: {agt1 > agt2}")

assert agt1 == agt2
assert agt1 > agt2
```

Output:
```
agt1 == agt2: True
agt1 > agt2: True  # BUG: Should be False!
```

## Why This Is A Bug

The `@functools.total_ordering` decorator expects classes to implement comparison operators that satisfy the mathematical properties of a total ordering:

1. **Irreflexivity of >**: For any object `a`, `a > a` must be `False`
2. **Antisymmetry**: If `a >= b` and `b >= a`, then `a == b`

The current implementation violates both:
- `agt > agt` returns `True` (violates irreflexivity)
- `agt1 > agt2` and `agt2 > agt1` are both `True`, even though `agt1 == agt2` (violates antisymmetry)

This is problematic because:
- It breaks the contract expected by `@functools.total_ordering`
- It violates user expectations for comparison operators
- It could cause unexpected behavior in sorting algorithms and ordered containers
- The same bug affects `AlwaysLessThan`

## Fix

```diff
--- a/xarray/core/dtypes.py
+++ b/xarray/core/dtypes.py
@@ -17,7 +17,8 @@ from xarray.core import utils
 @functools.total_ordering
 class AlwaysGreaterThan:
     def __gt__(self, other):
-        return True
+        if isinstance(other, type(self)):
+            return False
+        return True

     def __eq__(self, other):
         return isinstance(other, type(self))
@@ -26,7 +27,8 @@ class AlwaysGreaterThan:
 @functools.total_ordering
 class AlwaysLessThan:
     def __lt__(self, other):
-        return True
+        if isinstance(other, type(self)):
+            return False
+        return True

     def __eq__(self, other):
         return isinstance(other, type(self))
```