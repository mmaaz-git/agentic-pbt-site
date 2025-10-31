# Bug Report: pandas.qcut with duplicates='drop' Crashes on Skewed Data

**Target**: `pandas.qcut`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`pd.qcut()` crashes with a confusing error when using `duplicates='drop'` on data with many duplicate values, instead of gracefully handling duplicate bin edges as documented.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import pandas as pd
from hypothesis import given, strategies as st, settings, assume
from hypothesis.extra.pandas import series


@settings(max_examples=100)
@given(series(elements=st.floats(allow_nan=False, allow_infinity=False,
                                   min_value=-1e6, max_value=1e6), dtype=float))
def test_qcut_preserves_all_elements(s):
    assume(len(s) >= 4)
    assume(s.nunique() >= 4)

    q = 4
    result = pd.qcut(s, q=q, duplicates='drop')

    assert len(result) == len(s)
```

**Failing input**:
```python
s = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 2.225074e-313])
```

## Reproducing the Bug

```python
import pandas as pd

s = pd.Series([0.0] * 6 + [1.0, 0.5, 2.225074e-313])

pd.qcut(s, q=4, duplicates='drop')
```

**Output:**
```
ValueError: missing values must be missing in the same location both left and right sides
```

## Why This Is A Bug

The `duplicates='drop'` parameter is explicitly documented to handle cases where "bin edges are not unique". When quantiles fall on duplicate values (common with skewed data), the function should either:
1. Successfully create bins by dropping duplicates, or
2. Provide a clear, actionable error message

Instead, it crashes deep in the IntervalArray validation with a confusing error message that doesn't explain the root cause (duplicate quantile boundaries due to skewed data).

## Fix

The bug occurs in the bin edge dropping logic. When duplicate bin edges are dropped, it creates intervals where NaN appears in mismatched positions in the left and right arrays.

A proper fix would:
1. Detect when dropping duplicates would create invalid intervals
2. Either handle this case correctly or raise a clear error like: `ValueError: Cannot create quantiles - too many duplicate values. Try fewer quantiles or use pd.cut() instead.`