# Bug Report: pandas.core.methods.describe Quantiles Overflow and Violate Monotonicity for Extreme Int64 Values

**Target**: `pandas.core.methods.describe.describe_numeric_1d`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When `describe()` is called on a Series containing extreme int64 values, the quantiles violate mathematical monotonicity and can overflow beyond the maximum int64 value, producing impossible results.

## Property-Based Test

```python
import pandas as pd
from hypothesis import given, strategies as st, settings, assume, reproduce_failure
from hypothesis.extra.pandas import series


@given(series(dtype=int, index=st.just(pd.RangeIndex(0, 20))))
@settings(max_examples=200)
def test_describe_numeric_quantiles_sorted(s):
    assume(len(s.dropna()) > 0)

    result = s.describe()

    percentile_keys = ["25%", "50%", "75%"]
    percentiles = [result[k] for k in percentile_keys]

    for i in range(len(percentiles) - 1):
        assert percentiles[i] <= percentiles[i + 1], f"Quantiles not sorted: {percentile_keys[i]}={percentiles[i]} > {percentile_keys[i+1]}={percentiles[i+1]}"


# Test with the specific failing case directly
def test_specific_failing_case():
    failing_series = pd.Series([
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

    result = failing_series.describe()
    percentile_keys = ["25%", "50%", "75%"]
    percentiles = [result[k] for k in percentile_keys]

    for i in range(len(percentiles) - 1):
        assert percentiles[i] <= percentiles[i + 1], f"Quantiles not sorted: {percentile_keys[i]}={percentiles[i]} > {percentile_keys[i+1]}={percentiles[i+1]}"


if __name__ == "__main__":
    # First test with the specific failing case
    try:
        test_specific_failing_case()
        print("Test with specific case passed")
    except AssertionError as e:
        print(f"Test failed with error: {e}")
        print(f"Failing series reproduced the bug!")

    # Try property-based test
    print("\nRunning property-based test...")
    try:
        test_describe_numeric_quantiles_sorted()
        print("Property-based tests passed")
    except Exception as e:
        print(f"Property test found failing case: {e}")
```

<details>

<summary>
**Failing input**: Series with extreme negative values near -2.6e18 and positive values at int64 maximum (9.223372036854775807)
</summary>
```
Test failed with error: Quantiles not sorted: 25%=1.152921504606847e+19 > 50%=9.223372036854776e+18
Failing series reproduced the bug!

Running property-based test...
Property-based tests passed
```
</details>

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

<details>

<summary>
Output demonstrating non-monotonic quantiles and overflow
</summary>
```
25% quantile: 1.152921504606847e+19
50% quantile: 9.223372036854776e+18
75% quantile: 9.223372036854776e+18
Are quantiles sorted? False
```
</details>

## Why This Is A Bug

This bug violates two fundamental mathematical properties:

1. **Quantile Monotonicity Violation**: By definition, quantiles must be monotonically non-decreasing. The 25th percentile MUST be less than or equal to the 50th percentile, which MUST be less than or equal to the 75th percentile. In this case, the 25th percentile (1.152921504606847e+19) is greater than both the 50th and 75th percentiles (9.223372036854776e+18), which is mathematically impossible.

2. **Value Range Violation**: The computed 25th percentile (1.152921504606847e+19) exceeds the maximum int64 value (9.223372036854775807e+18). Since all input values are valid int64 integers within the range [-2.6e18, 9.223372036854775807e+18], it is impossible for any quantile to exceed this maximum input value.

The bug occurs during the conversion of int64 values to float64 for interpolation in the quantile calculation. When extreme int64 values are processed, precision is lost and arithmetic operations can overflow, leading to these impossible results.

## Relevant Context

- The issue manifests in `pandas.core.methods.describe.describe_numeric_1d` at line 234 where `series.quantile(percentiles).tolist()` is called
- The underlying `Series.quantile()` method performs float64 conversion that loses precision with extreme int64 values
- This affects any code using `describe()` on Series with large integer values, including:
  - Unix timestamps in nanoseconds (common in time series data)
  - Large ID values in databases
  - Financial data with high precision requirements

The pandas documentation for `describe()` states it provides "descriptive statistics" but doesn't warn about precision loss with extreme int64 values. The function should either handle these values correctly or raise a warning when precision loss could occur.

## Proposed Fix

The issue requires fixing the underlying quantile calculation to handle extreme int64 values without overflow. Here's a high-level approach:

```diff
# In pandas/core/methods/describe.py, around line 217-236
def describe_numeric_1d(series: Series, percentiles: Sequence[float]) -> Series:
    """Describe series containing numerical data.

    Parameters
    ----------
    series : Series
        Series to be described.
    percentiles : list-like of numbers
        The percentiles to include in the output.
    """
    from pandas import Series

    formatted_percentiles = format_percentiles(percentiles)

    stat_index = ["count", "mean", "std", "min"] + formatted_percentiles + ["max"]
+
+   # Check for extreme int64 values that might overflow in float64 conversion
+   if series.dtype == np.int64:
+       max_val = series.max()
+       min_val = series.min()
+       # Check if values are too extreme for accurate float64 representation
+       if abs(max_val - min_val) > 2**53:  # float64 mantissa precision limit
+           # Use higher precision or alternative quantile calculation
+           # Option 1: Use numpy's percentile with 'lower' interpolation to avoid float overflow
+           quantile_values = []
+           for p in percentiles:
+               q_val = np.percentile(series.dropna().values, p * 100, method='lower')
+               quantile_values.append(float(q_val))
+       else:
+           quantile_values = series.quantile(percentiles).tolist()
+   else:
+       quantile_values = series.quantile(percentiles).tolist()
+
    d = (
        [series.count(), series.mean(), series.std(), series.min()]
-       + series.quantile(percentiles).tolist()
+       + quantile_values
        + [series.max()]
    )
```

A more comprehensive fix would involve updating the `Series.quantile()` method itself to detect and handle extreme int64 values, potentially using:
- Integer-based quantile calculation for integer dtypes when possible
- Higher precision arithmetic (e.g., numpy.float128 or decimal) for interpolation
- Warnings when precision loss is detected