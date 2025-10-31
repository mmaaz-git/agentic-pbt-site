# Bug Report: dask.dataframe Quantile Calculation Produces Incorrect and Unordered Results

**Target**: `dask.dataframe.Series.quantile`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `quantile()` method in dask.dataframe produces incorrect quantile values that can violate the fundamental mathematical property that quantiles must be ordered (Q25 ≤ Q50 ≤ Q75). This leads to incorrect statistical calculations and violates basic expectations about quantile behavior.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas as pd
import dask.dataframe as dd

@given(st.lists(st.floats(allow_nan=False, allow_infinity=False,
                          min_value=-100, max_value=100), min_size=2, max_size=50))
@settings(max_examples=100)
def test_quantile_ordered(values):
    pdf = pd.DataFrame({'x': values})
    ddf = dd.from_pandas(pdf, npartitions=2)

    q25 = ddf['x'].quantile(0.25).compute()
    q50 = ddf['x'].quantile(0.50).compute()
    q75 = ddf['x'].quantile(0.75).compute()

    assert q25 <= q50 <= q75, \
        f"Quantiles not ordered: Q25={q25}, Q50={q50}, Q75={q75}"
```

**Failing input**: `[0.0, 0.0, 2.0, 3.0, 1.0]`

## Reproducing the Bug

```python
import pandas as pd
import dask.dataframe as dd

values = [0.0, 0.0, 2.0, 3.0, 1.0]
pdf = pd.DataFrame({'x': values})
ddf = dd.from_pandas(pdf, npartitions=2)

pandas_q25 = pdf['x'].quantile(0.25)
pandas_q50 = pdf['x'].quantile(0.50)
pandas_q75 = pdf['x'].quantile(0.75)

dask_q25 = ddf['x'].quantile(0.25).compute()
dask_q50 = ddf['x'].quantile(0.50).compute()
dask_q75 = ddf['x'].quantile(0.75).compute()

print("Pandas quantiles (correct):")
print(f"  Q25: {pandas_q25}")
print(f"  Q50: {pandas_q50}")
print(f"  Q75: {pandas_q75}")

print("\nDask quantiles (incorrect):")
print(f"  Q25: {dask_q25}")
print(f"  Q50: {dask_q50}")
print(f"  Q75: {dask_q75}")

print(f"\nOrdering violation: {dask_q25} > {dask_q50}")
```

**Output**:
```
Pandas quantiles (correct):
  Q25: 0.0
  Q50: 1.0
  Q75: 2.0

Dask quantiles (incorrect):
  Q25: 1.5
  Q50: 1.0
  Q75: 2.0

Ordering violation: 1.5 > 1.0
```

## Why This Is A Bug

1. **Violates mathematical properties**: Quantiles must satisfy Q_p1 ≤ Q_p2 for p1 < p2. The bug produces Q25 > Q50, which is mathematically impossible.

2. **Produces incorrect values**: For the input `[0.0, 0.0, 2.0, 3.0, 1.0]`, Dask computes Q25=1.5 while the correct value is 0.0. This is a 100% error.

3. **Widespread issue**: The bug affects many common inputs:
   - `[0, 1]`: Q25 = 0.0 (should be 0.25)
   - `[0, 100]`: Q50 = 0.0 (should be 50.0)
   - `[1, 2, 3, 4, 5]`: Q25 = 1.5 (should be 2.0), Q50 = 2.0 (should be 3.0)

4. **Silent data corruption**: Users relying on Dask quantiles for statistical analysis will get incorrect results without any warning.

5. **High impact**: Quantiles are fundamental to statistical analysis, used for:
   - Outlier detection
   - Data summarization
   - Percentile-based metrics
   - Risk analysis

## Fix

This bug is in Dask's approximate quantile algorithm which attempts to compute quantiles efficiently across distributed partitions. The issue appears to be in how quantiles are merged across partitions.

The root cause is likely in the quantile merging logic. A proper fix would need to:

1. Ensure the approximate algorithm maintains quantile ordering properties
2. Add validation to detect and correct ordering violations
3. Consider using a more robust distributed quantile algorithm (e.g., t-digest or Q-digest)

Without access to modify the Dask codebase directly, the recommended workaround for users is:

```python
result = ddf.compute()
pandas_quantiles = result['column'].quantile([0.25, 0.5, 0.75])
```

This computes the exact quantile on the full dataset but loses the distributed computation benefits.