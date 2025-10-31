# Bug Report: BooleanArray any()/all() Violate Kleene Logic

**Target**: `pandas.arrays.BooleanArray.any()` and `pandas.arrays.BooleanArray.all()`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

BooleanArray's docstring states it "implements Kleene logic (sometimes called three-value logic)" but the `any()` and `all()` methods violate Kleene logic when all values are NA or when mixing known values with NA.

## Property-Based Test

```python
import pandas.arrays as pa
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings


@settings(max_examples=200)
@given(st.integers(min_value=1, max_value=20))
def test_all_na_any_should_be_na(n):
    arr = pa.BooleanArray(np.zeros(n, dtype='bool'),
                          np.ones(n, dtype='bool'))

    any_result = arr.any()
    assert pd.isna(any_result), f"any() on all-NA array should return NA per Kleene logic, but got {any_result}"


@settings(max_examples=200)
@given(st.integers(min_value=1, max_value=20))
def test_all_na_all_should_be_na(n):
    arr = pa.BooleanArray(np.zeros(n, dtype='bool'),
                          np.ones(n, dtype='bool'))

    all_result = arr.all()
    assert pd.isna(all_result), f"all() on all-NA array should return NA per Kleene logic, but got {all_result}"
```

**Failing input**: `n=1` (any array of all-NA values)

## Reproducing the Bug

```python
import pandas.arrays as pa
import numpy as np
import pandas as pd

arr_all_na = pa.BooleanArray(np.array([False], dtype='bool'),
                              np.array([True], dtype='bool'))

print(f"all-NA array: {arr_all_na}")
print(f"any(): {arr_all_na.any()}")
print(f"all(): {arr_all_na.all()}")
print(f"\nExpected: both should be pd.NA")
print(f"Actual any() is NA: {pd.isna(arr_all_na.any())}")
print(f"Actual all() is NA: {pd.isna(arr_all_na.all())}")

arr_false_na = pa.BooleanArray(np.array([False, False], dtype='bool'),
                                np.array([False, True], dtype='bool'))
print(f"\n[False, NA]:")
print(f"  any(): {arr_false_na.any()} (should be NA)")
print(f"  all(): {arr_false_na.all()} (should be False)")

arr_true_na = pa.BooleanArray(np.array([True, False], dtype='bool'),
                               np.array([False, True], dtype='bool'))
print(f"\n[True, NA]:")
print(f"  any(): {arr_true_na.any()} (should be True)")
print(f"  all(): {arr_true_na.all()} (should be NA)")
```

Output:
```
all-NA array: <BooleanArray>
[<NA>]
Length: 1, dtype: boolean
any(): False
all(): True

Expected: both should be pd.NA
Actual any() is NA: False
Actual all() is NA: False

[False, NA]:
  any(): False (should be NA)
  all(): False (should be False)

[True, NA]:
  any(): True (should be True)
  all(): True (should be NA)
```

## Why This Is A Bug

BooleanArray's docstring explicitly states: "BooleanArray implements Kleene logic (sometimes called three-value logic) for logical operations."

In Kleene logic, when the truth value is unknown (NA), the result should be NA unless it can be determined regardless of the NA value:

- `any([NA])` should return NA (we don't know if any are True)
- `all([NA])` should return NA (we don't know if all are True)
- `any([False, NA])` should return NA (depends on whether NA is True)
- `all([True, NA])` should return NA (depends on whether NA is True)
- `any([True, ...])` correctly returns True (at least one is definitely True)
- `all([False, ...])` correctly returns False (at least one is definitely False)

The current implementation incorrectly treats all-NA arrays as if they were all-False for `any()` and all-True for `all()`, which violates three-valued logic semantics.

## Fix

The `any()` and `all()` methods should return `pd.NA` when the result depends on NA values. Here's the logic that should be implemented:

```python
def any(self):
    if self._mask.all():
        return pd.NA

    if self._data[~self._mask].any():
        return True

    if self._mask.any():
        return pd.NA

    return False

def all(self):
    if self._mask.all():
        return pd.NA

    if (~self._data[~self._mask]).any():
        return False

    if self._mask.any():
        return pd.NA

    return True
```