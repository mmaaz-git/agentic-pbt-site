# Bug Report: pandas.core.window.Rolling.var() Returns Negative Variance

**Target**: `pandas.core.window.Rolling.var()`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `rolling().var()` method returns negative variance values due to numerical instability, violating the fundamental mathematical property that variance must always be non-negative.

## Property-Based Test

```python
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings, assume


@given(
    series=st.lists(
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
        min_size=5, max_size=50
    ),
    window=st.integers(min_value=2, max_value=10)
)
@settings(max_examples=200)
def test_rolling_var_non_negative(series, window):
    assume(len(series) >= window)

    s = pd.Series(series)
    rolling_var = s.rolling(window=window).var()

    valid_mask = ~rolling_var.isna()
    if valid_mask.sum() > 0:
        assert (rolling_var[valid_mask] >= -1e-10).all(), \
            f"Variance should be non-negative, got {rolling_var[valid_mask].min()}"

# Run the test
if __name__ == "__main__":
    try:
        test_rolling_var_non_negative()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")
        # Re-run with the specific failing case to show it
        print("\nDemonstrating with the minimal failing case:")
        series = [0.0, 494699.5, 0.0, 0.0, 0.00390625]
        window = 3
        s = pd.Series(series)
        rolling_var = s.rolling(window=window).var()
        print(f"Series: {series}")
        print(f"Window: {window}")
        print(f"Rolling variance values: {rolling_var.tolist()}")
        print(f"Minimum variance: {rolling_var.min()}")
        print(f"Is minimum variance negative? {rolling_var.min() < 0}")
```

<details>

<summary>
**Failing input**: `series=[0.0, 494699.5, 0.0, 0.0, 0.00390625], window=3`
</summary>
```
Test failed: Variance should be non-negative, got -1.5258787243510596e-05

Demonstrating with the minimal failing case:
Series: [0.0, 494699.5, 0.0, 0.0, 0.00390625]
Window: 3
Rolling variance values: [nan, nan, 81575865100.08333, 81575865100.08333, -1.017252596587544e-05]
Minimum variance: -1.017252596587544e-05
Is minimum variance negative? True
```
</details>

## Reproducing the Bug

```python
import pandas as pd
import numpy as np

series = [0.0, 494699.5, 0.0, 0.0, 0.00390625]
s = pd.Series(series)

rolling_var = s.rolling(window=3).var()
negative_var = rolling_var.iloc[4]

print(f"Rolling variance at index 4: {negative_var}")
print(f"Window data: {series[2:5]}")
print(f"Expected (numpy): {np.var(series[2:5], ddof=1)}")
print(f"BUG: Variance is negative: {negative_var < 0}")

# Additional validation
print(f"\nAll rolling variance values:")
for i, val in enumerate(rolling_var):
    if not pd.isna(val):
        print(f"  Index {i}: {val}")

# Manual calculation to verify
window_data = series[2:5]
mean = sum(window_data) / len(window_data)
squared_diffs = [(x - mean)**2 for x in window_data]
manual_var = sum(squared_diffs) / (len(window_data) - 1)  # ddof=1
print(f"\nManual calculation:")
print(f"  Mean: {mean}")
print(f"  Squared differences: {squared_diffs}")
print(f"  Variance (ddof=1): {manual_var}")
```

<details>

<summary>
Pandas returns negative variance (-1.017e-05) instead of positive variance (5.086e-06)
</summary>
```
Rolling variance at index 4: -1.017252596587544e-05
Window data: [0.0, 0.0, 0.00390625]
Expected (numpy): 5.086263020833334e-06
BUG: Variance is negative: True

All rolling variance values:
  Index 2: 81575865100.08333
  Index 3: 81575865100.08333
  Index 4: -1.017252596587544e-05

Manual calculation:
  Mean: 0.0013020833333333333
  Squared differences: [1.6954210069444444e-06, 1.6954210069444444e-06, 6.781684027777779e-06]
  Variance (ddof=1): 5.086263020833334e-06
```
</details>

## Why This Is A Bug

This violates the fundamental mathematical definition of variance. Variance is defined as the average of squared deviations from the mean: `Var(X) = E[(X - μ)²]`. Since it involves squaring differences, variance must always be non-negative (≥ 0).

The bug occurs when the rolling window contains values with vastly different magnitudes. In this case, the window transitions from containing the large value 494699.5 to containing only small values [0.0, 0.0, 0.00390625]. This creates numerical instability in the incremental variance calculation algorithm used by pandas.

The pandas implementation returns `-1.017e-05` (negative) while the correct value is `+5.086e-06` (positive). This is not just a sign error - the magnitude is also incorrect by a factor of ~2. This error can:
- Cause crashes in dependent calculations (e.g., `std = sqrt(var)` would fail with domain error)
- Silently corrupt statistical analyses that depend on variance
- Propagate errors through financial risk models, machine learning algorithms, and scientific computations

## Relevant Context

This is a known issue in pandas, documented in multiple GitHub issues (#52407, #1090, #10242, #42064). The pandas team has acknowledged this as a "clear bug" that stems from numerical instability in the Cython implementation at `pandas/_libs/window/aggregations.pyx`.

The rolling variance implementation uses an incremental algorithm for efficiency, but this algorithm suffers from catastrophic cancellation when dealing with values of vastly different magnitudes. The issue affects not just variance but also other rolling statistics like skewness, kurtosis, and correlation.

Pandas documentation: https://pandas.pydata.org/docs/reference/api/pandas.core.window.rolling.Rolling.var.html
Related GitHub issues: https://github.com/pandas-dev/pandas/issues/52407

## Proposed Fix

The fix requires implementing a numerically stable variance algorithm in the Cython backend. The current incremental algorithm should be replaced with either Welford's online algorithm or a two-pass algorithm with compensated summation. Since this requires modifying Cython code in pandas internals, here's a high-level approach:

1. Locate the variance calculation in `pandas/_libs/window/aggregations.pyx`
2. Replace the current incremental algorithm with Welford's method which maintains numerical stability
3. Add a safeguard to clamp any negative values to 0 due to floating-point errors
4. Add regression tests for this specific case

The Welford algorithm maintains running sums of deviations and squared deviations in a way that avoids catastrophic cancellation:
- M_n = M_{n-1} + (x_n - M_{n-1})/n  (running mean)
- S_n = S_{n-1} + (x_n - M_{n-1})(x_n - M_n)  (sum of squared deviations)
- Variance = S_n / (n-1) for sample variance

This approach is numerically stable even with values of vastly different magnitudes.