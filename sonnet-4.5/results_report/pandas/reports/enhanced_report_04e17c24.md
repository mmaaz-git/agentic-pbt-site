# Bug Report: pandas.core.window.rolling.Rolling.var Returns Negative Variance Values

**Target**: `pandas.core.window.rolling.Rolling.var`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The rolling variance calculation in pandas returns mathematically impossible negative values, violating the fundamental property that variance must always be non-negative. This occurs due to numerical instability when processing windows containing large values followed by much smaller values.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas as pd

@given(
    st.lists(
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10),
        min_size=2,
        max_size=20
    ),
    st.integers(min_value=2, max_value=10)
)
@settings(max_examples=1000)
def test_rolling_variance_always_nonnegative(data, window):
    if window > len(data):
        return
    s = pd.Series(data)
    var = s.rolling(window=window).var()
    valid = var[~var.isna()]
    assert (valid >= 0).all(), f"Found negative variance: {valid[valid < 0]}"

# Run the test
if __name__ == "__main__":
    try:
        test_rolling_variance_always_nonnegative()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")
        # Try the specific failing example
        print("\nTrying specific failing example:")
        data = [3222872787.0, 0.0, 2.0, 0.0]
        window = 3
        s = pd.Series(data)
        var = s.rolling(window=window).var()
        valid = var[~var.isna()]
        print(f"Data: {data}")
        print(f"Window: {window}")
        print(f"Rolling variance: {var.tolist()}")
        print(f"Valid variances: {valid.tolist()}")
        negative_vars = valid[valid < 0]
        if len(negative_vars) > 0:
            print(f"Negative variances found: {negative_vars.tolist()}")
```

<details>

<summary>
**Failing input**: `data=[3222872787.0, 0.0, 2.0, 0.0], window=3`
</summary>
```
Test failed: Found negative variance: 3   -6.666667
dtype: float64

Trying specific failing example:
Data: [3222872787.0, 0.0, 2.0, 0.0]
Window: 3
Rolling variance: [nan, nan, 3.462302998246467e+18, -511.66666682561237]
Valid variances: [3.462302998246467e+18, -511.66666682561237]
Negative variances found: [-511.66666682561237]
```
</details>

## Reproducing the Bug

```python
import pandas as pd
import numpy as np

# Minimal test case that demonstrates negative variance
data = [3222872787.0, 0.0, 2.0, 0.0]
s = pd.Series(data)
var = s.rolling(window=3).var()

print("Input data:", data)
print("\nRolling variance (window=3):")
print(var)
print("\nDetailed analysis:")
for i in range(len(var)):
    if pd.notna(var.iloc[i]):
        if i >= 2:
            window_data = data[i-2:i+1]
            print(f"Index {i}: window={window_data}, variance={var.iloc[i]:.6f}")

            # Manual calculation for verification
            mean = sum(window_data) / len(window_data)
            manual_var = sum((x - mean)**2 for x in window_data) / (len(window_data) - 1)
            print(f"  Manual calculation: mean={mean:.6f}, variance={manual_var:.6f}")

            # NumPy calculation for comparison
            numpy_var = np.var(window_data, ddof=1)
            print(f"  NumPy calculation: variance={numpy_var:.6f}")

            if var.iloc[i] < 0:
                print(f"  ❌ NEGATIVE VARIANCE DETECTED: {var.iloc[i]:.6f}")
```

<details>

<summary>
Output shows negative variance of -511.67 for window [0.0, 2.0, 0.0]
</summary>
```
Input data: [3222872787.0, 0.0, 2.0, 0.0]

Rolling variance (window=3):
0             NaN
1             NaN
2    3.462303e+18
3   -5.116667e+02
dtype: float64

Detailed analysis:
Index 2: window=[3222872787.0, 0.0, 2.0], variance=3462302998246467072.000000
  Manual calculation: mean=1074290929.666667, variance=3462302998246466560.000000
  NumPy calculation: variance=3462302998246466560.000000
Index 3: window=[0.0, 2.0, 0.0], variance=-511.666667
  Manual calculation: mean=0.666667, variance=1.333333
  NumPy calculation: variance=1.333333
  ❌ NEGATIVE VARIANCE DETECTED: -511.666667
```
</details>

## Why This Is A Bug

Variance is mathematically defined as the expected value of squared deviations from the mean: Var(X) = E[(X - μ)²]. Since squared values are always non-negative and the average of non-negative values must be non-negative, **variance can never be negative**.

The pandas documentation states that `rolling.var()` calculates the rolling variance with formula `(sum of squared deviations) / (N - ddof)`. For the window `[0.0, 2.0, 0.0]`:
- Expected variance: 1.333 (as calculated manually and by NumPy)
- Actual pandas result: -511.667

This violates the fundamental mathematical property of variance and produces silently incorrect results. The error appears to stem from numerical instability in the variance algorithm when a rolling window transitions from containing very large values (3.2 billion) to much smaller values (0, 2). The implementation likely uses the computationally unstable formula `Var(X) = E[X²] - E[X]²` which suffers from catastrophic cancellation when E[X²] and E[X]² are close in magnitude.

## Relevant Context

The bug manifests in pandas 2.3.2 and likely affects earlier versions. The issue occurs in the Cython implementation of rolling aggregations (`pandas._libs.window.aggregations`).

Key observations:
- NumPy correctly computes the same variance as 1.333
- The error magnitude is severe (off by ~513, not just rounding errors)
- No warnings or errors are raised, leading to silent data corruption
- This affects real-world use cases like financial data analysis where values can vary by orders of magnitude

Pandas documentation: https://pandas.pydata.org/docs/reference/api/pandas.core.window.rolling.Rolling.var.html
Related numerical stability issues: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

## Proposed Fix

The fix requires implementing a numerically stable variance algorithm. The current implementation likely needs modification in the Cython code. Here's a high-level approach using Welford's online algorithm which is numerically stable:

```python
# Pseudocode for numerically stable rolling variance
def stable_rolling_variance(values, window_size):
    n = len(values)
    result = []

    for i in range(n):
        if i < window_size - 1:
            result.append(np.nan)
        else:
            window = values[i - window_size + 1:i + 1]

            # Welford's algorithm for numerical stability
            mean = 0.0
            M2 = 0.0

            for j, x in enumerate(window, 1):
                delta = x - mean
                mean += delta / j
                delta2 = x - mean
                M2 += delta * delta2

            variance = M2 / (window_size - 1)  # ddof=1
            result.append(variance)

    return result
```

The actual fix would need to be implemented in the Cython code at `pandas/_libs/window/aggregations.pyx`, replacing the current variance calculation with either Welford's algorithm or a two-pass algorithm that computes the mean first, then the sum of squared deviations.