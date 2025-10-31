# Bug Report: pandas qcut Crashes on Edge Case With Duplicate Values

**Target**: `pandas.qcut`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`pd.qcut` crashes with a ValueError when given a Series with mostly identical values and one very small different value, even when using `duplicates='drop'`.

## Property-Based Test

```python
import pandas as pd
from hypothesis import assume, given, settings, strategies as st
from hypothesis.extra import pandas as pdst


@given(
    pdst.series(dtype=float, elements=st.floats(allow_nan=False, allow_infinity=False, min_value=0, max_value=100)),
    st.integers(min_value=2, max_value=10)
)
@settings(max_examples=500)
def test_qcut_handles_edge_cases(s, q):
    assume(len(s) >= q)
    assume(s.notna().all())
    assume(len(s.unique()) >= q)

    result = pd.qcut(s, q=q, duplicates='drop')
    assert len(result) == len(s)
```

**Failing input**: `pd.Series([0.0, 0.0, 2.225074e-311])` with `q=2`

## Reproducing the Bug

```python
import pandas as pd

s = pd.Series([0.0, 0.0, 2.225074e-311])

result = pd.qcut(s, q=2, duplicates='drop')
```

Output:
```
ValueError: missing values must be missing in the same location both left and right sides
```

## Why This Is A Bug

The function `qcut` is designed to bin values into quantiles. When given the `duplicates='drop'` parameter, it should handle cases where quantile edges result in duplicate bins. However, with certain edge cases involving very small differences in values, the internal IntervalArray validation fails because it produces intervals where the left bound exists but the right bound is NaN. This should either be handled gracefully by the `duplicates='drop'` logic or result in a more informative error message.

## Fix

The issue is in the quantile calculation and interval construction logic. When the data has extreme clustering with tiny differences, the quantile calculation produces invalid interval boundaries. The fix should either:

1. Better handle edge cases in the quantile calculation to avoid producing NaN boundaries
2. Improve the `duplicates='drop'` logic to handle these malformed intervals
3. Provide a more informative error message explaining the issue

The root cause appears to be in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/reshape/tile.py` around line 483 in `_bins_to_cuts` where labels are formatted.