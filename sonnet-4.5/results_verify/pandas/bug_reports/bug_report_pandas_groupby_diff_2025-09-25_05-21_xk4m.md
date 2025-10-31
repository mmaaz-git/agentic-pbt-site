# Bug Report: pandas.core.groupby diff() Precision Loss

**Target**: `pandas.core.groupby.DataFrameGroupBy.diff()` and `pandas.core.groupby.SeriesGroupBy.diff()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `diff()` method on grouped data produces different results than `diff()` on ungrouped data for large integer values (> 2^53). This occurs because grouped `diff()` converts values to float64 before computing differences, while ungrouped `diff()` computes differences in int64 before converting to float64, leading to different precision loss patterns.

## Property-Based Test

```python
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings


@given(
    st.integers(min_value=2**53, max_value=2**60),
    st.integers(min_value=-1000, max_value=1000),
)
@settings(max_examples=100)
def test_grouped_diff_matches_ungrouped_diff(large_val, small_val):
    df = pd.DataFrame({
        'group': ['a', 'a'],
        'value': [large_val, small_val]
    })

    grouped_diff = df.groupby('group')['value'].diff()
    ungrouped_diff = df['value'].diff()

    assert ungrouped_diff.loc[1] == grouped_diff.loc[1], \
        f"Grouped diff ({grouped_diff.loc[1]}) != ungrouped diff ({ungrouped_diff.loc[1]}) for values {large_val}, {small_val}"
```

**Failing input**: `large_val=9007199254768175, small_val=1`

## Reproducing the Bug

```python
import pandas as pd

df = pd.DataFrame({
    'group': ['a', 'a'],
    'value': [9007199254768175, 1]
})

grouped_diff = df.groupby('group')['value'].diff()
ungrouped_diff = df['value'].diff()

print(f"Grouped diff():   {grouped_diff.loc[1]}")
print(f"Ungrouped diff(): {ungrouped_diff.loc[1]}")
print(f"Match: {grouped_diff.loc[1] == ungrouped_diff.loc[1]}")
```

**Output:**
```
Grouped diff():   -9007199254768176.0
Ungrouped diff(): -9007199254768174.0
Match: False
```

The difference is 2.0, which represents a precision error.

## Why This Is A Bug

The `diff()` method should produce consistent results regardless of whether it's called on grouped or ungrouped data. The grouped version is converting to float64 before computing the difference, causing precision loss on the large integer value (9007199254768175 → 9007199254768176.0), which then produces an incorrect difference.

The ungrouped version correctly computes `1 - 9007199254768175 = -9007199254768174` in int64, then converts to float64, preserving the correct result.

Users expect `df.groupby('group')['value'].diff()` to behave identically to `df['value'].diff()` when there's only one group, but this bug violates that expectation for large integers.

## Fix

The fix would be to ensure that grouped `diff()` computes differences in the original dtype (int64) before converting to float64, matching the behavior of ungrouped `diff()`. The conversion to float64 should happen after the subtraction, not before.

This likely requires modifying the implementation in `pandas/core/groupby/ops.py` or `pandas/core/groupby/groupby.py` where the diff operation is applied within groups.