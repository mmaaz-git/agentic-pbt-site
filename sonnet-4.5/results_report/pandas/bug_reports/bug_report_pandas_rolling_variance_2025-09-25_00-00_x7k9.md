# Bug Report: pandas.core.window Rolling Variance Returns Negative Values and Inconsistent with Std

**Target**: `pandas.core.window.rolling.Rolling.var` and `pandas.core.window.rolling.Rolling.std`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `Rolling.var()` method can return negative variance values and produces results inconsistent with `Rolling.std()` due to numerical precision issues when processing data with large magnitude differences. Mathematically, variance must always be non-negative and must equal the square of standard deviation.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume, settings
import pandas as pd
import math

@given(
    values=st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=50
    ),
    window=st.integers(min_value=2, max_value=10)
)
@settings(max_examples=500)
def test_rolling_variance_equals_std_squared(values, window):
    assume(window <= len(values))

    s = pd.Series(values)
    rolling = s.rolling(window=window)

    rolling_var = rolling.var()
    rolling_std = rolling.std()

    for i in range(len(s)):
        if not pd.isna(rolling_var.iloc[i]) and not pd.isna(rolling_std.iloc[i]):
            expected_var = rolling_std.iloc[i] ** 2
            assert math.isclose(rolling_var.iloc[i], expected_var, rel_tol=1e-9, abs_tol=1e-9), \
                f"At index {i}: var {rolling_var.iloc[i]} != std^2 {expected_var}"
```

**Failing input**: `values=[131073.0, 1e-05, 0.0], window=2`

## Reproducing the Bug

```python
import pandas as pd
import numpy as np

values = [131073.0, 1e-05, 0.0]
s = pd.Series(values)
rolling_var = s.rolling(window=2).var()
rolling_std = s.rolling(window=2).std()

print("Rolling variance:", rolling_var.values)
print("Rolling std^2:  ", (rolling_std ** 2).values)

print("\nAt index 2 (window: [1e-05, 0.0]):")
print(f"  Pandas rolling variance: {rolling_var.iloc[2]}")
print(f"  Pandas rolling std^2:    {rolling_std.iloc[2] ** 2}")
print(f"  NumPy variance (correct): {np.var([1e-05, 0.0], ddof=1)}")

print("\nAdditional example:")
values2 = [1e-05, -978956.7043791538, 0.0, 0.015625]
s2 = pd.Series(values2)
rolling_var2 = s2.rolling(window=2).var()
print(f"Rolling variance: {rolling_var2.values}")
```

Output:
```
Rolling variance: [            nan  8.59006566e+09 -1.90729863e-06]
Rolling std^2:   [           nan  8.59006566e+09  0.00000000e+00]

At index 2 (window: [1e-05, 0.0]):
  Pandas rolling variance: -1.9072986328125e-06
  Pandas rolling std^2:    0.0
  NumPy variance (correct): 5.000000000000001e-11

Additional example:
Rolling variance: [ nan  4.79178135e+11  4.79178135e+11 -6.10351581e-05]
```

## Why This Is A Bug

This bug violates two fundamental mathematical properties:

1. **Variance must be non-negative**: Variance is defined as the average of squared deviations from the mean. Since it involves squaring, variance **must always be ≥ 0**.

2. **Variance equals std²**: By definition, variance is the square of standard deviation. When `rolling.var() ≠ (rolling.std())²`, the implementation is internally inconsistent.

The bug manifests when:
1. A large value is processed in a rolling window
2. Followed by much smaller values in subsequent windows
3. The rolling variance calculation uses an incremental algorithm that accumulates catastrophic cancellation errors

Impact:
- Returns mathematically impossible negative variance values
- `var()` and `std()` produce inconsistent results
- Affects data analysis, financial calculations, quality control, and any application relying on rolling statistics
- Silent corruption of statistical results (no error/warning raised)

## Fix

This bug likely stems from using Welford's online variance algorithm or similar incremental computation without proper numerical safeguards. The issue is in the Cython implementation of the rolling variance calculation.

A fix would involve:
1. Adding numerical stability checks to prevent catastrophic cancellation
2. Using a more stable variance algorithm (e.g., two-pass or compensated summation)
3. Adding a post-computation check to clamp variance to be non-negative (min 0.0)

Since the actual implementation is in Cython (`pandas/_libs/window/aggregations.pyx`), a detailed patch would require examining that code. A minimal safeguard would be:

```python
# In the rolling variance calculation
result_var = computed_variance
if result_var < 0:
    result_var = 0.0  # Clamp negative values to zero
```

However, the proper fix should address the root numerical stability issue rather than just clamping the output.