# Bug Report: pandas Rolling Mean Precision Loss with Subnormal Numbers

**Target**: `pandas.core.window.rolling.Rolling.mean` (via `pandas.api.typing.Rolling`)
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`Series.rolling().mean()` produces severely incorrect results when processing very small floating-point numbers near the subnormal range, with errors approaching 100%.

## Property-Based Test

```python
import numpy as np
import pandas as pd
from hypothesis import given, assume
import hypothesis.extra.pandas as pdst
from hypothesis import strategies as st


@given(pdst.series(elements=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
                   index=pdst.range_indexes(min_size=10, max_size=100)),
       st.integers(min_value=2, max_value=10))
def test_rolling_mean_bounds(series, window_size):
    assume(len(series) >= window_size)

    rolling_mean = series.rolling(window=window_size).mean()

    for i in range(window_size - 1, len(series)):
        if not np.isnan(rolling_mean.iloc[i]):
            window_data = series.iloc[i - window_size + 1:i + 1]
            assert rolling_mean.iloc[i] >= window_data.min()
            assert rolling_mean.iloc[i] <= window_data.max()
```

**Failing input**: Window containing `[-2.798597e-225, -2.225074e-308]`

## Reproducing the Bug

```python
import pandas as pd

series = pd.Series([
    1.000000e+00,
    1.605551e-178,
    -2.798597e-225,
    -2.225074e-308,
    -2.798597e-225,
])

rolling_mean = series.rolling(window=2).mean()

print(f"Window at index 3: {series.iloc[2:4].values}")
print(f"Expected mean: {series.iloc[2:4].mean()}")
print(f"Rolling mean: {rolling_mean.iloc[3]}")
print(f"Relative error: {abs(rolling_mean.iloc[3] - series.iloc[2:4].mean()) / abs(series.iloc[2:4].mean())}")
```

Output:
```
Window at index 3: [-2.798597e-225 -2.225074e-308]
Expected mean: -1.3992985e-225
Rolling mean: -1.112537e-308
Relative error: 0.9999999999999999
```

The rolling mean is off by ~100%, computing -1.1125e-308 instead of -1.3993e-225.

## Why This Is A Bug

1. **Silent data corruption**: The function returns incorrect numerical results without any warning
2. **Violates mathematical invariant**: The mean of values must lie between the minimum and maximum
3. **100% relative error**: The computed value is essentially unrelated to the correct answer
4. **Affects scientific computing**: Users working with physical simulations or scientific data near float64 limits will get wrong results

The bug appears when:
- Processing a longer series (not just two values)
- Values include numbers near the subnormal threshold (~2.2e-308)
- The rolling window passes through regions with very small numbers

## Fix

The issue likely stems from how pandas accumulates running sums in the rolling window implementation. The bug appears in the compiled rolling mean code (possibly in Cython or C extensions).

The fix would require:
1. Using Kahan summation or compensated summation for better numerical stability
2. Detecting when values are in the subnormal range and using higher precision arithmetic
3. Recomputing from scratch when precision loss is detected rather than using incremental updates

The implementation is likely in `pandas/_libs/window/aggregations.pyx` or similar compiled extension code.