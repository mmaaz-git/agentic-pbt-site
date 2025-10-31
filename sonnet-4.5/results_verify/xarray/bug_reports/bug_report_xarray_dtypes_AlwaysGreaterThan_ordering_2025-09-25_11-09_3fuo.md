# Bug Report: xarray.core.dtypes.AlwaysGreaterThan Ordering Violation

**Target**: `xarray.core.dtypes.AlwaysGreaterThan` and `xarray.core.dtypes.AlwaysLessThan`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `AlwaysGreaterThan` and `AlwaysLessThan` sentinel classes violate the mathematical irreflexivity property of strict ordering. Specifically, `AlwaysGreaterThan() > AlwaysGreaterThan()` returns `True` when it should return `False`, and similarly for `AlwaysLessThan`. This breaks fundamental ordering invariants and can cause inconsistent sorting behavior.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from xarray.core.dtypes import AlwaysGreaterThan, AlwaysLessThan

@given(st.one_of(st.integers(), st.floats(allow_nan=False, allow_infinity=False)))
@settings(max_examples=200)
def test_always_greater_than_irreflexivity(value):
    always_gt = AlwaysGreaterThan()

    # Irreflexivity: x > x should always be False
    assert not (always_gt > always_gt), "AlwaysGreaterThan violates irreflexivity"

    # But it should still be > any other value
    assert always_gt > value

@given(st.one_of(st.integers(), st.floats(allow_nan=False, allow_infinity=False)))
@settings(max_examples=200)
def test_always_less_than_irreflexivity(value):
    always_lt = AlwaysLessThan()

    # Irreflexivity: x < x should always be False
    assert not (always_lt < always_lt), "AlwaysLessThan violates irreflexivity"

    # But it should still be < any other value
    assert always_lt < value
```

**Failing input**: Any instance of `AlwaysGreaterThan` or `AlwaysLessThan`

## Reproducing the Bug

```python
from xarray.core.dtypes import AlwaysGreaterThan, AlwaysLessThan

# Bug 1: AlwaysGreaterThan violates irreflexivity
always_gt = AlwaysGreaterThan()
assert always_gt > always_gt  # True - BUG! Should be False
assert always_gt == always_gt  # True - correct

# Bug 2: AlwaysLessThan violates irreflexivity
always_lt = AlwaysLessThan()
assert always_lt < always_lt  # True - BUG! Should be False
assert always_lt == always_lt  # True - correct

# Demonstrates antisymmetry violation
always_gt1 = AlwaysGreaterThan()
always_gt2 = AlwaysGreaterThan()
assert always_gt1 > always_gt2  # True
assert always_gt2 > always_gt1  # True
assert always_gt1 == always_gt2  # True
```

## Why This Is A Bug

The mathematical properties of strict ordering require **irreflexivity**: for any value `x`, the relation `x > x` must be `False`. Similarly, `x < x` must be `False`. These classes violate this fundamental property.

From a practical perspective:
1. **Sorting inconsistency**: Having both `x > y` and `y > x` return `True` while `x == y` also returns `True` violates antisymmetry
2. **Unexpected comparison results**: Code that relies on standard ordering semantics (e.g., `not (x > x)`) will behave incorrectly
3. **Violates Python's data model**: The comparison operators should follow standard mathematical conventions

## Fix

```diff
--- a/xarray/core/dtypes.py
+++ b/xarray/core/dtypes.py
@@ -123,7 +123,7 @@ def maybe_promote(dtype: np.dtype) -> tuple[np.dtype, Any]:
 @functools.total_ordering
 class AlwaysGreaterThan:
     def __gt__(self, other):
-        return True
+        return not isinstance(other, type(self))

     def __eq__(self, other):
         return isinstance(other, type(self))
@@ -132,7 +132,7 @@ class AlwaysGreaterThan:
 @functools.total_ordering
 class AlwaysLessThan:
     def __lt__(self, other):
-        return True
+        return not isinstance(other, type(self))

     def __eq__(self, other):
         return isinstance(other, type(self))
```

This fix ensures that:
- `AlwaysGreaterThan() > AlwaysGreaterThan()` returns `False` (correct)
- `AlwaysGreaterThan() == AlwaysGreaterThan()` returns `True` (correct)
- `AlwaysGreaterThan() > other_value` returns `True` (correct)
- Similarly for `AlwaysLessThan`