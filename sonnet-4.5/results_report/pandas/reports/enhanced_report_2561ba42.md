# Bug Report: pandas.core.window.Rolling.mean Catastrophic Precision Error Violating Mathematical Invariants

**Target**: `pandas.core.window.Rolling.mean`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The pandas rolling mean calculation produces mathematically impossible results when the data contains very large numbers, returning a mean value that exceeds the maximum value in the window by 14 orders of magnitude due to accumulated floating-point precision errors.

## Property-Based Test

```python
import math

import pandas as pd
from hypothesis import assume, given, settings, strategies as st


@given(
    st.lists(
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10),
        min_size=1,
        max_size=100,
    ),
    st.integers(min_value=1, max_value=10),
)
@settings(max_examples=500)
def test_rolling_min_max_bounds(data, window):
    assume(window <= len(data))
    s = pd.Series(data)

    rolling_min = s.rolling(window=window).min()
    rolling_max = s.rolling(window=window).max()
    rolling_mean = s.rolling(window=window).mean()

    for i in range(window - 1, len(data)):
        min_val = rolling_min.iloc[i]
        max_val = rolling_max.iloc[i]
        mean_val = rolling_mean.iloc[i]

        assert min_val <= mean_val or math.isclose(min_val, mean_val, rel_tol=1e-9, abs_tol=1e-9), \
            f"At {i}: rolling_min {min_val} > rolling_mean {mean_val}"
        assert mean_val <= max_val or math.isclose(mean_val, max_val, rel_tol=1e-9, abs_tol=1e-9), \
            f"At {i}: rolling_mean {mean_val} > rolling_max {max_val}"
```

<details>

<summary>
**Failing input**: `data=[1.0, -4294967297.0, 0.99999, 0.0, 0.0, 1.6675355247098508e-21], window=3`
</summary>
```
Running property-based test...
Testing with the specific failing input:
data=[1.0, -4294967297.0, 0.99999, 0.0, 0.0, 1.6675355247098508e-21]
window=3
TEST FAILED: At 5: rolling_mean 1.5894571940104224e-07 > rolling_max 1.6675355247098508e-21

Running full hypothesis test suite...
Test failed: Hypothesis found 2 distinct failures. (2 sub-exceptions)
```
</details>

## Reproducing the Bug

```python
import pandas as pd

data = [1.0, -4294967297.0, 0.99999, 0.0, 0.0, 1.6675355247098508e-21]
s = pd.Series(data)
rolling_mean = s.rolling(window=3).mean()
rolling_min = s.rolling(window=3).min()
rolling_max = s.rolling(window=3).max()

window_at_5 = data[3:6]
expected_mean = sum(window_at_5) / 3
actual_mean = rolling_mean.iloc[5]

print(f"Data: {data}")
print(f"\nWindow at index 5: {window_at_5}")
print(f"Expected mean: {expected_mean}")
print(f"Pandas rolling mean: {actual_mean}")
print(f"Pandas rolling min: {rolling_min.iloc[5]}")
print(f"Pandas rolling max: {rolling_max.iloc[5]}")
print(f"\nBug: mean ({actual_mean}) > max ({max(window_at_5)})")
print(f"Mathematical violation: min <= mean <= max is FALSE")
print(f"Relative error: {abs(actual_mean - expected_mean) / abs(expected_mean) * 100:.2e}%")
```

<details>

<summary>
Output showing mathematical invariant violation
</summary>
```
Data: [1.0, -4294967297.0, 0.99999, 0.0, 0.0, 1.6675355247098508e-21]

Window at index 5: [0.0, 0.0, 1.6675355247098508e-21]
Expected mean: 5.558451749032836e-22
Pandas rolling mean: 1.5894571940104224e-07
Pandas rolling min: 0.0
Pandas rolling max: 1.6675355247098508e-21

Bug: mean (1.5894571940104224e-07) > max (1.6675355247098508e-21)
Mathematical violation: min <= mean <= max is FALSE
Relative error: 2.86e+16%
```
</details>

## Why This Is A Bug

This violates the fundamental mathematical invariant that the arithmetic mean of any finite set of numbers must lie between the minimum and maximum values. The pandas rolling mean function returns 1.589e-07 for a window containing [0.0, 0.0, 1.67e-21], where the maximum value is 1.67e-21. This is mathematically impossible - the mean cannot exceed the maximum by 14 orders of magnitude.

The bug manifests when:
1. The series contains very large numbers (here -4294967297 ≈ -2^32)
2. The rolling window slides past these large values to much smaller values
3. Floating-point precision errors from the large values contaminate subsequent calculations
4. The contamination persists even after the large value exits the window

This represents a silent data corruption issue where pandas returns completely incorrect results (relative error of 2.86×10^16%) without any warning or error, potentially leading to catastrophically wrong analytical conclusions in scientific or financial applications.

## Relevant Context

The pandas documentation for `rolling().mean()` states it "Calculate the rolling mean" but makes no mention of numerical precision limitations or potential violations of mathematical properties. Users reasonably expect that basic mathematical invariants (min ≤ mean ≤ max) will hold for all finite, non-NaN inputs.

This appears related to known pandas issues with numerical stability in rolling calculations:
- GitHub issue #37051: discusses precision issues in rolling operations
- GitHub issue #19308: addresses floating-point accumulation errors

The bug likely stems from the incremental update algorithm used in the Cython implementation of rolling calculations, where accumulated floating-point errors from operations on values with vastly different magnitudes corrupt the running sum/count used to compute the mean.

## Proposed Fix

The fix requires modifying the underlying Cython implementation to use a more numerically stable algorithm. Since the actual implementation is in compiled Cython code, here's a high-level approach:

1. Implement Kahan summation or a similar compensated summation algorithm to reduce floating-point accumulation errors
2. Add range checking to detect when values span extreme magnitudes and switch to a more robust algorithm
3. Consider resetting accumulators when the window completely slides past values with extreme magnitudes
4. Add warnings when numerical instability is detected (e.g., when mean falls outside min/max bounds)

For immediate mitigation, users working with extreme value ranges should:
- Normalize or scale data before applying rolling operations
- Use higher precision dtypes (e.g., np.float128) where available
- Implement custom rolling mean calculations for critical applications requiring guaranteed numerical stability