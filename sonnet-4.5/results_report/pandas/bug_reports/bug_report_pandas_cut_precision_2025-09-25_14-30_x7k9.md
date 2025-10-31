# Bug Report: pandas.core.reshape.tile.cut Precision Causes Values Outside Bin Boundaries

**Target**: `pandas.core.reshape.tile.cut`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `pd.cut()` function with default `precision=3` can assign values to bins that don't actually contain them. When bin boundaries are rounded for display/storage, values near the boundary may fall outside their assigned bin's interval.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume, settings
import pandas as pd

@settings(max_examples=500)
@given(
    values=st.lists(
        st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
        min_size=10,
        max_size=100
    ),
    n_bins=st.integers(min_value=2, max_value=20)
)
def test_cut_respects_bin_boundaries(values, n_bins):
    assume(len(set(values)) >= 2)

    result = pd.cut(values, bins=n_bins)

    for i, (val, cat) in enumerate(zip(values, result)):
        if pd.notna(cat):
            left, right = cat.left, cat.right
            assert left < val <= right, \
                f"Value {val} at index {i} not in its assigned bin {cat}"
```

**Failing input**: `values=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.15625], n_bins=2`

## Reproducing the Bug

```python
import pandas as pd

values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.15625]
result = pd.cut(values, bins=2, precision=3)

val = 1.15625
cat = result[9]

print(f"Value: {val}")
print(f"Assigned interval: {cat}")
print(f"Value in interval? {val in cat}")
```

Output:
```
Value: 1.15625
Assigned interval: (0.578, 1.156]
Value in interval? False
```

The value `1.15625` is assigned to the interval `(0.578, 1.156]`, but `1.15625 > 1.156`, so it violates the interval boundary.

## Why This Is A Bug

The documentation states that `cut()` bins values into discrete intervals with specific boundaries. When `precision=3` (the default), the bin edges are rounded to 3 decimal places, which can cause the upper bound to be less than the maximum value being binned. This violates the fundamental property that a value should be contained within its assigned bin.

The issue occurs because:
1. Pandas extends the range by 0.1% to ensure all values fit
2. With the extended range, bin edges are computed
3. These edges are then rounded to `precision` decimal places
4. The rounding can cause the upper bin boundary to become less than the maximum value
5. Values are then assigned to bins that don't mathematically contain them

## Fix

The precision parameter should only affect the display of intervals, not their actual boundaries used for value assignment. The bins should be computed and assigned using full floating-point precision, and only rounded for display purposes.

Alternatively, when rounding bin boundaries, ensure that:
- The minimum bin edge is always `<=` the minimum value
- The maximum bin edge is always `>=` the maximum value

One approach would be to use `np.floor` for the minimum and `np.ceil` for the maximum when applying precision rounding, rather than simple rounding.