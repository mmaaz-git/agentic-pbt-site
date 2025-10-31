# Bug Report: pandas.core.methods.describe Quantiles Not Sorted for Extreme Int64 Values

**Target**: `pandas.core.methods.describe.describe_numeric_1d`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When `describe()` is called on a Series containing extreme int64 values, the computed quantiles are not sorted in ascending order. Specifically, the 25% quantile can be greater than the 50% and 75% quantiles, violating the fundamental property that quantiles must be monotonically increasing.

## Property-Based Test

```python
import pandas as pd
from hypothesis import given, strategies as st, settings, assume
from hypothesis.extra.pandas import series


@given(series(dtype=int, index=st.just(pd.RangeIndex(0, 20))))
@settings(max_examples=200)
def test_describe_numeric_quantiles_sorted(s):
    assume(len(s.dropna()) > 0)

    result = s.describe()

    percentile_keys = ["25%", "50%", "75%"]
    percentiles = [result[k] for k in percentile_keys]

    for i in range(len(percentiles) - 1):
        assert percentiles[i] <= percentiles[i + 1]
```

**Failing input**: A Series with extreme negative values near the int64 minimum and positive values at the int64 maximum.

## Reproducing the Bug

```python
import pandas as pd

s = pd.Series([
    -2601355999077536245,
    -2601355999077536245,
    -2601355999077536240,
    -1,
    -2601355999077536240,
    9223372036854775807,
    9223372036854775807,
    9223372036854775807,
    9223372036854775807,
    9223372036854775807,
    9223372036854775807,
    9223372036854775807,
    9223372036854775807,
    9223372036854775807,
    9223372036854775807,
    9223372036854775807,
    9223372036854775807,
    9223372036854775807,
    9223372036854775807,
    9223372036854775807,
], dtype='int64')

result = s.describe()

print(f"25% quantile: {result['25%']}")
print(f"50% quantile: {result['50%']}")
print(f"75% quantile: {result['75%']}")
print(f"Are quantiles sorted? {result['25%'] <= result['50%'] <= result['75%']}")
```

**Output:**
```
25% quantile: 1.152921504606847e+19
50% quantile: 9.223372036854776e+18
75% quantile: 9.223372036854776e+18
Are quantiles sorted? False
```

## Why This Is A Bug

Quantiles (percentiles) by definition must be monotonically increasing. The 25th percentile should always be less than or equal to the 50th percentile, which should be less than or equal to the 75th percentile. In this case:

- 25% quantile: 1.152921504606847e+19
- 50% quantile: 9.223372036854776e+18

The 25% quantile is larger than the 50% quantile, which is mathematically impossible and indicates a numerical overflow or precision error.

Furthermore, the computed 25% quantile (1.152921504606847e+19) exceeds the maximum int64 value (9.223372036854775807), which is impossible given that all input values are valid int64 integers.

## Fix

The root cause is in the underlying quantile calculation used by `describe_numeric_1d`. The issue occurs when converting int64 values to float64 for interpolation - extreme int64 values near the boundaries lose precision and can overflow during arithmetic operations.

A high-level fix would involve:
1. Detecting when int64 values are too extreme for accurate float64 conversion
2. Using higher precision arithmetic (e.g., float128 or decimal) for quantile calculation in these cases
3. Alternatively, implementing integer-based quantile calculation for integer dtypes when possible

The bug manifests in `describe_numeric_1d` at line 234:
```python
+ series.quantile(percentiles).tolist()
```

The underlying `Series.quantile()` method needs to handle extreme int64 values without overflow.
