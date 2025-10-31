# Bug Report: pandas.core.ops kleene_and/kleene_or/kleene_xor RecursionError

**Target**: `pandas.core.ops.kleene_and`, `pandas.core.ops.kleene_or`, `pandas.core.ops.kleene_xor`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The kleene logic functions (`kleene_and`, `kleene_or`, `kleene_xor`) in `pandas.core.ops` cause infinite recursion and crash with a `RecursionError` when called with scalar boolean arguments and `None` masks.

## Property-Based Test

```python
from pandas.core import ops
from hypothesis import given, strategies as st

@given(st.booleans(), st.booleans())
def test_kleene_and_without_mask_equals_regular_and(a, b):
    result = ops.kleene_and(a, b, None, None)
    expected = a and b
    assert result == expected
```

**Failing input**: `a=False, b=True` (or any combination of scalar booleans)

## Reproducing the Bug

```python
from pandas.core import ops

result = ops.kleene_and(False, True, None, None)
```

**Output:**
```
RecursionError: maximum recursion depth exceeded
```

The same bug occurs with `kleene_or` and `kleene_xor`:

```python
ops.kleene_or(False, True, None, None)
ops.kleene_xor(False, True, None, None)
```

## Why This Is A Bug

1. **Crashes on valid inputs**: The function signature and docstring indicate that the functions accept scalar booleans (`left, right : ndarray, NA, or bool`), but calling them with scalar booleans causes a crash.

2. **Infinite recursion**: The bug is in the argument swapping logic. When `left_mask is None`, the function swaps arguments and recursively calls itself. However, if BOTH masks are `None` (as with two scalar booleans), the swapped arguments still have `left_mask=None`, causing infinite recursion.

3. **Affects all three functions**: `kleene_and`, `kleene_or`, and `kleene_xor` all have the same bug pattern.

## Root Cause

Located in `/pandas/core/ops/mask_ops.py`, around line 157 (kleene_and):

```python
def kleene_and(...):
    if left_mask is None:
        return kleene_and(right, left, right_mask, left_mask)
    ...
```

When both `left_mask` and `right_mask` are `None`, this creates infinite recursion because after swapping, `left_mask` (which was `right_mask = None`) is still `None`, triggering another swap.

## Fix

Add a check for the case when both masks are `None`:

```diff
def kleene_and(
    left: bool | libmissing.NAType | np.ndarray,
    right: bool | libmissing.NAType | np.ndarray,
    left_mask: np.ndarray | None,
    right_mask: np.ndarray | None,
):
    # To reduce the number of cases, we ensure that `left` & `left_mask`
    # always come from an array, not a scalar. This is safe, since
    # A & B == B & A
+   if left_mask is None and right_mask is None:
+       # Both are scalars, handle directly without recursion
+       raise TypeError("kleene_and requires at least one array argument")
    if left_mask is None:
        return kleene_and(right, left, right_mask, left_mask)
    ...
```

Alternatively, if scalar-scalar operations should be supported, implement direct handling:

```diff
+   if left_mask is None and right_mask is None:
+       # Both are scalars, compute directly
+       if isinstance(left, type(libmissing.NA)) or isinstance(right, type(libmissing.NA)):
+           return libmissing.NA, None
+       return left & right, None
    if left_mask is None:
        return kleene_and(right, left, right_mask, left_mask)
```

The same fix applies to `kleene_or` and `kleene_xor`.