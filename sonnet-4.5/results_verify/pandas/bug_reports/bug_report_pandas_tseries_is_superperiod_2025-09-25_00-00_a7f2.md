# Bug Report: pandas.tseries.frequencies Logical Impossibility in is_superperiod

**Target**: `pandas.tseries.frequencies.is_superperiod` and `pandas.tseries.frequencies.is_subperiod`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `is_superperiod` function incorrectly returns True for both `is_superperiod('D', 'B')` and `is_superperiod('B', 'D')`, violating the mathematical constraint that two frequencies cannot both be superperiods of each other.

## Property-Based Test

```python
import pandas.tseries.frequencies as freq_module
from hypothesis import given, strategies as st, settings

PERIOD_FREQUENCIES = ['D', 'W', 'M', 'Q', 'Y', 'h', 'min', 's', 'ms', 'B', 'BM', 'BQ', 'BY']

@given(
    source=st.sampled_from(PERIOD_FREQUENCIES),
    target=st.sampled_from(PERIOD_FREQUENCIES),
)
@settings(max_examples=200)
def test_is_subperiod_superperiod_symmetry(source, target):
    is_sub = freq_module.is_subperiod(source, target)
    is_super = freq_module.is_superperiod(target, source)

    assert is_sub == is_super, (
        f"Symmetry violated: is_subperiod({source}, {target})={is_sub} but "
        f"is_superperiod({target}, {source})={is_super}"
    )
```

**Failing input**: `source='D', target='B'`

## Reproducing the Bug

```python
import pandas.tseries.frequencies as freq_module

source, target = 'D', 'B'

is_sub_dt = freq_module.is_subperiod(source, target)
is_super_dt = freq_module.is_superperiod(source, target)
is_sub_td = freq_module.is_subperiod(target, source)
is_super_td = freq_module.is_superperiod(target, source)

print(f"is_subperiod('D', 'B') = {is_sub_dt}")
print(f"is_superperiod('D', 'B') = {is_super_dt}")
print(f"is_subperiod('B', 'D') = {is_sub_td}")
print(f"is_superperiod('B', 'D') = {is_super_td}")

assert not (is_super_dt and is_super_td), "Both cannot be superperiods of each other!"
```

Output:
```
is_subperiod('D', 'B') = False
is_superperiod('D', 'B') = True
is_subperiod('B', 'D') = False
is_superperiod('B', 'D') = True
AssertionError: Both cannot be superperiods of each other!
```

## Why This Is A Bug

According to the docstrings:
- `is_subperiod(source, target)` returns True if downsampling is possible from source to target
- `is_superperiod(source, target)` returns True if upsampling is possible from source to target

These functions should satisfy a symmetry property: if `is_subperiod(A, B)` returns True (downsampling from A to B is possible), then `is_superperiod(B, A)` must also return True (upsampling from B to A is possible).

However, for daily ('D') and business day ('B') frequencies:
- `is_superperiod('D', 'B') = True`
- `is_superperiod('B', 'D') = True`

This is logically impossible - both frequencies cannot be superperiods of each other. One must be a subperiod or they must be incomparable.

## Fix

The issue likely stems from how business day frequencies are handled in the period comparison logic. The functions need to correctly model the relationship between regular calendar frequencies ('D') and business day frequencies ('B'). These are not strictly comparable in a hierarchical sense (business days are a subset of calendar days, but not a regular sampling).

A high-level fix would involve:
1. Identifying that 'D' and 'B' are incomparable in the strict sub/super period hierarchy
2. Returning False for both `is_subperiod('D', 'B')` and `is_superperiod('D', 'B')`
3. Similarly returning False for both `is_subperiod('B', 'D')` and `is_superperiod('B', 'D')`