# Bug Report: pandas.api.typing.NAType Comparison Inconsistency

**Target**: `pandas.api.typing.NAType` (pd.NA)
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

NA comparisons (`==` and `!=`) return `bool` instead of `NAType` when comparing with container types (lists, dicts, tuples, sets), None, and generic objects. This violates the documented three-valued logic semantics where all comparisons should return NA.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas as pd
from pandas.api.typing import NAType


@given(st.lists(st.integers()) | st.dictionaries(st.integers(), st.integers()))
def test_na_equality_consistency_with_containers(container):
    result = pd.NA == container
    assert isinstance(result, NAType), (
        f"Expected NA == {type(container).__name__} to return NAType, "
        f"got {type(result).__name__}: {result}"
    )


def test_na_equality_consistency_with_none():
    result = pd.NA == None
    assert isinstance(result, NAType), (
        f"Expected NA == None to return NAType, got {type(result).__name__}: {result}"
    )
```

**Failing input**: Any container (e.g., `[]`, `{}`, `()`, `set()`), or `None`

## Reproducing the Bug

```python
import pandas as pd

result_none = pd.NA == None
result_list = pd.NA == []
result_int = pd.NA == 0

print(f"NA == None: {result_none} (type: {type(result_none).__name__})")
print(f"NA == []: {result_list} (type: {type(result_list).__name__})")
print(f"NA == 0: {result_int} (type: {type(result_int).__name__})")
```

**Output:**
```
NA == None: False (type: bool)
NA == []: False (type: bool)
NA == 0: <NA> (type: NAType)
```

**Expected:** All three comparisons should return `NAType` (NA), not `bool`.

**Affected comparisons:**
- `pd.NA == None` → Returns `False` (should return `NA`)
- `pd.NA != None` → Returns `True` (should return `NA`)
- `pd.NA == []` → Returns `False` (should return `NA`)
- `pd.NA == {}` → Returns `False` (should return `NA`)
- `pd.NA == ()` → Returns `False` (should return `NA`)
- `pd.NA == set()` → Returns `False` (should return `NA`)
- `pd.NA == object()` → Returns `False` (should return `NA`)

**Working correctly:**
- `pd.NA == 0` → Returns `NA` ✓
- `pd.NA == ""` → Returns `NA` ✓
- `pd.NA == True` → Returns `NA` ✓

## Why This Is A Bug

The NAType docstring and examples demonstrate that comparisons should follow three-valued logic, where `NA == x` always returns `NA`:

```python
>>> pd.NA == pd.NA
<NA>
```

This behavior is fundamental to NA's semantics as a proper missing value indicator. The current implementation inconsistently returns `False` for container types and `None`, violating this invariant. This could lead to:

1. **Unexpected behavior in conditional logic**: `if pd.NA == None:` evaluates to `if False:` instead of raising a TypeError
2. **Data integrity issues**: Missing value propagation fails when comparing with None or containers
3. **Inconsistent user experience**: Users cannot rely on NA comparisons behaving uniformly

## Fix

The bug appears to be in the `__eq__` and `__ne__` methods of NAType, which likely have special-case handling for container types and None. The fix would be to remove these special cases and ensure all comparisons return NAType consistently.

Without access to the C implementation, a high-level fix would ensure that:
1. `NAType.__eq__` always returns `pd.NA` (the singleton NAType instance)
2. `NAType.__ne__` always returns `pd.NA` (the singleton NAType instance)
3. No special-casing for specific types (None, containers, etc.)

The implementation should follow the principle that **any comparison involving NA returns NA**, regardless of the other operand's type.