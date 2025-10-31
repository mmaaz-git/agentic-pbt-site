# Bug Report: pandas.cut duplicates='drop' Crashes on Near-Duplicate Edges

**Target**: `pandas.cut()` and `pandas.qcut()`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`pd.cut()` with `duplicates='drop'` crashes with a confusing error when applied to data with very small range. The error message initially suggests using `duplicates='drop'` to handle duplicate bin edges, but using that parameter causes a crash with an unrelated error about "missing values must be missing in the same location".

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import pandas as pd
from hypothesis import given, strategies as st, settings


@given(
    x=st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10), min_size=5, max_size=50),
    bins=st.integers(min_value=2, max_value=10)
)
@settings(max_examples=300, deadline=None)
def test_cut_preserves_length(x, bins):
    result = pd.cut(x, bins=bins, duplicates='drop')
    assert len(result) == len(x)
```

**Failing input**: `x=[0.0, 0.0, 0.0, 0.0, 5e-324]` with `bins=2`

## Reproducing the Bug

```python
import pandas as pd

x = [0.0, 0.0, 0.0, 0.0, 5e-324]

try:
    result = pd.cut(x, bins=2)
except ValueError as e:
    print(f"Error: {e}")
```

Output:
```
Error: Bin edges must be unique: Index([0.0, 0.0, 5e-324], dtype='float64').
You can drop duplicate edges by setting the 'duplicates' kwarg
```

Following the error message's advice:
```python
result = pd.cut(x, bins=2, duplicates='drop')
```

Output:
```
ValueError: missing values must be missing in the same location both left and right sides
```

The same issue affects `pd.qcut()`:
```python
x = [0.0]*9 + [2.225073858507e-311]
pd.qcut(x, q=2, duplicates='drop')
```
Also raises: `ValueError: missing values must be missing in the same location both left and right sides`

## Why This Is A Bug

The `duplicates='drop'` parameter is designed to handle duplicate bin edges. However:

1. **Misleading guidance**: The error message tells users to use `duplicates='drop'`, but doing so causes a different crash
2. **Unhelpful error**: The second error "missing values must be missing in the same location" is confusing and unrelated to the actual problem
3. **Unexpected failure**: `duplicates='drop'` should gracefully handle near-duplicate edges, not crash

The issue occurs when:
- Data has very small variance (e.g., mostly zeros with one tiny value like `5e-324`)
- Computed bin edges include near-duplicates
- After dropping duplicates, the resulting intervals cannot form a valid IntervalArray

This affects real-world use cases:
- Data with very small variance
- Values near machine epsilon
- Scientific computing with subnormal floats

## Fix

The bug is in `pandas/core/reshape/tile.py` in the `_bins_to_cuts()` function. After dropping duplicate edges, the code should:

1. Validate that sufficient unique edges remain to create valid intervals
2. Provide a clear, actionable error message if not
3. Consider a more robust binning strategy for edge cases

Suggested fix:
```python
# After dropping duplicate edges in _bins_to_cuts
if len(unique_edges) < 2:
    raise ValueError(
        f"After dropping duplicate edges, insufficient unique bins remain. "
        f"Data has very small range: min={x.min()}, max={x.max()}. "
        f"Consider using fewer bins or checking if data has sufficient variance."
    )
```

Alternatively, fall back to a single-bin strategy when this occurs, rather than crashing.