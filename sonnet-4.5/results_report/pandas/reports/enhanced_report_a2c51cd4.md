# Bug Report: pandas.core.window.rolling.Rolling.mean Produces Mathematically Impossible Results with Subnormal Float64 Numbers

**Target**: `pandas.core.window.rolling.Rolling.mean`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The pandas rolling mean function produces completely incorrect results (100% relative error) when processing floating-point numbers near the subnormal range (~10^-308), violating the fundamental mathematical property that a mean must fall between the minimum and maximum values of its inputs.

## Property-Based Test

```python
import numpy as np
import pandas as pd
from hypothesis import given, assume, settings
import hypothesis.extra.pandas as pdst
from hypothesis import strategies as st


@given(pdst.series(elements=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
                   index=pdst.range_indexes(min_size=10, max_size=100)),
       st.integers(min_value=2, max_value=10))
@settings(max_examples=1000)
def test_rolling_mean_bounds(series, window_size):
    assume(len(series) >= window_size)

    rolling_mean = series.rolling(window=window_size).mean()

    for i in range(window_size - 1, len(series)):
        if not np.isnan(rolling_mean.iloc[i]):
            window_data = series.iloc[i - window_size + 1:i + 1]
            assert rolling_mean.iloc[i] >= window_data.min(), \
                f"Mean {rolling_mean.iloc[i]} is below min {window_data.min()} at index {i}"
            assert rolling_mean.iloc[i] <= window_data.max(), \
                f"Mean {rolling_mean.iloc[i]} is above max {window_data.max()} at index {i}"

if __name__ == "__main__":
    test_rolling_mean_bounds()
```

<details>

<summary>
**Failing input**: Window containing values near 10^-225 and 10^-308
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/36/hypo.py", line 26, in <module>
  |     test_rolling_mean_bounds()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/36/hypo.py", line 9, in test_rolling_mean_bounds
  |     index=pdst.range_indexes(min_size=10, max_size=100)),
  |
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/36/hypo.py", line 22, in test_rolling_mean_bounds
    |     assert rolling_mean.iloc[i] <= window_data.max(), \
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    | AssertionError: Mean 3.125321196232664e-250 is above max 1.171214450183335e-300 at index 6
    | Falsifying example: test_rolling_mean_bounds(
    |     series=0    -1.565510e+00
    |     1    -4.857500e+04
    |     2    -1.000000e+06
    |     3   -9.375964e-250
    |     4    1.171214e-300
    |     5     0.000000e+00
    |     6     0.000000e+00
    |     7     0.000000e+00
    |     8     0.000000e+00
    |     9     0.000000e+00
    |     dtype: float64,
    |     window_size=3,
    | )
    | Explanation:
    |     These lines were always and only run by failing examples:
    |         /home/npc/pbt/agentic-pbt/worker_/36/hypo.py:23
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/36/hypo.py", line 20, in test_rolling_mean_bounds
    |     assert rolling_mean.iloc[i] >= window_data.min(), \
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    | AssertionError: Mean 1.002977898476931e-229 is below min 2.005955796953862e-229 at index 3
    | Falsifying example: test_rolling_mean_bounds(
    |     series=0     1.000000e+06
    |     1     4.857700e+04
    |     2     2.394996e-85
    |     3    2.005956e-229
    |     4     4.857700e+04
    |     5     4.857700e+04
    |     6     4.857700e+04
    |     7     4.857700e+04
    |     8     4.857700e+04
    |     9     4.857700e+04
    |     dtype: float64,
    |     window_size=2,
    | )
    | Explanation:
    |     These lines were always and only run by failing examples:
    |         /home/npc/pbt/agentic-pbt/worker_/36/hypo.py:21
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
import pandas as pd
import numpy as np

series = pd.Series([
    1.000000e+00,
    1.605551e-178,
    -2.798597e-225,
    -2.225074e-308,
    -2.798597e-225,
])

rolling_mean = series.rolling(window=2).mean()

print("Series values:")
for i, val in enumerate(series):
    print(f"  Index {i}: {val:e}")

print("\nRolling mean with window=2:")
for i, val in enumerate(rolling_mean):
    if not np.isnan(val):
        print(f"  Index {i}: {val:e}")

print("\n--- Focus on problematic window at index 3 ---")
print(f"Window at index 3: {series.iloc[2:4].values}")
print(f"  Min value in window: {series.iloc[2:4].min():e}")
print(f"  Max value in window: {series.iloc[2:4].max():e}")
print(f"  Expected mean (computed directly): {series.iloc[2:4].mean():e}")
print(f"  Rolling mean result: {rolling_mean.iloc[3]:e}")

expected_mean = series.iloc[2:4].mean()
actual_mean = rolling_mean.iloc[3]
print(f"\nError Analysis:")
print(f"  Absolute error: {abs(actual_mean - expected_mean):e}")
print(f"  Relative error: {abs(actual_mean - expected_mean) / abs(expected_mean):.15f}")

print(f"\nMathematical invariant check:")
min_val = series.iloc[2:4].min()
max_val = series.iloc[2:4].max()
if min_val <= actual_mean <= max_val:
    print(f"  ✓ Mean is within bounds [{min_val:e}, {max_val:e}]")
else:
    print(f"  ✗ VIOLATION: Mean {actual_mean:e} is OUTSIDE bounds [{min_val:e}, {max_val:e}]")

print("\n--- Also check window at index 4 ---")
print(f"Window at index 4: {series.iloc[3:5].values}")
print(f"  Expected mean: {series.iloc[3:5].mean():e}")
print(f"  Rolling mean result: {rolling_mean.iloc[4]:e}")
print(f"  Relative error: {abs(rolling_mean.iloc[4] - series.iloc[3:5].mean()) / abs(series.iloc[3:5].mean()):.15f}")
```

<details>

<summary>
Output showing mathematical invariant violation
</summary>
```
Series values:
  Index 0: 1.000000e+00
  Index 1: 1.605551e-178
  Index 2: -2.798597e-225
  Index 3: -2.225074e-308
  Index 4: -2.798597e-225

Rolling mean with window=2:
  Index 1: 5.000000e-01
  Index 2: 8.027755e-179
  Index 3: -1.112537e-308
  Index 4: 0.000000e+00

--- Focus on problematic window at index 3 ---
Window at index 3: [-2.798597e-225 -2.225074e-308]
  Min value in window: -2.798597e-225
  Max value in window: -2.225074e-308
  Expected mean (computed directly): -1.399298e-225
  Rolling mean result: -1.112537e-308

Error Analysis:
  Absolute error: 1.399298e-225
  Relative error: 1.000000000000000

Mathematical invariant check:
  ✗ VIOLATION: Mean -1.112537e-308 is OUTSIDE bounds [-2.798597e-225, -2.225074e-308]

--- Also check window at index 4 ---
Window at index 4: [-2.225074e-308 -2.798597e-225]
  Expected mean: -1.399298e-225
  Rolling mean result: 0.000000e+00
  Relative error: 1.000000000000000
```
</details>

## Why This Is A Bug

This violates the fundamental mathematical definition of arithmetic mean, which must always fall between the minimum and maximum values of the dataset. The bug manifests as:

1. **Mathematical Impossibility**: The computed mean (-1.112537e-308) falls outside the window's bounds [-2.798597e-225, -2.225074e-308], which is mathematically impossible for any valid mean calculation.

2. **100% Relative Error**: The rolling mean produces results that are completely wrong, not just imprecise. For the window at index 4, it even returns exactly 0.0 when the true mean is -1.399298e-225.

3. **Silent Data Corruption**: The function returns these incorrect values without any warning, error, or indication that the computation has failed.

4. **Violates IEEE 754 Semantics**: Subnormal numbers (values between ±2.225e-308 and ±4.94e-324) are valid IEEE 754 float64 values and should be handled correctly by numerical algorithms.

## Relevant Context

The pandas documentation acknowledges potential numerical imprecision in rolling operations, stating that "Some windowing aggregation, mean, sum, var and std methods may suffer from numerical imprecision due to the underlying windowing algorithms accumulating sums" and that "Kahan summation is used to compute the rolling sums to preserve accuracy as much as possible."

However, the observed behavior goes far beyond "imprecision" - it's a complete failure of the algorithm when dealing with subnormal numbers. The bug appears to be specific to:
- Series containing values near the subnormal threshold (≈10^-308 for float64)
- Rolling windows that transition through these extreme values
- The incremental update algorithm used for efficiency (simple 2-element series work correctly)

This impacts scientific computing applications that may legitimately work with extremely small values, such as:
- Quantum physics simulations dealing with probability amplitudes
- Numerical solutions to differential equations that approach zero
- Statistical analyses of rare events with very small probabilities

Documentation: https://pandas.pydata.org/docs/reference/api/pandas.core.window.rolling.Rolling.mean.html
Source location: pandas/_libs/window/aggregations.pyx (compiled Cython extension)

## Proposed Fix

The issue likely stems from catastrophic cancellation or underflow in the incremental sum update algorithm. While a detailed fix would require examining the Cython source, the high-level approach would be:

1. Detect when values approach the subnormal threshold
2. Switch from incremental updates to direct computation for affected windows
3. Implement compensated summation algorithms specifically designed for extreme values
4. Add guards against underflow to zero when accumulating very small values

A conceptual fix (not actual patch since the code is in compiled Cython):

```diff
# Pseudocode for the fix in the rolling sum accumulator
def update_rolling_sum(old_sum, add_value, remove_value):
    # Current implementation (simplified)
-   return old_sum + add_value - remove_value

    # Fixed implementation with underflow detection
+   # Check if we're dealing with subnormal range values
+   if abs(add_value) < 1e-300 or abs(remove_value) < 1e-300:
+       # Recompute from scratch to avoid accumulation errors
+       return compute_sum_from_window()
+   else:
+       # Use compensated summation for normal range
+       return kahan_sum(old_sum, add_value, -remove_value)
```

The actual implementation would need to be done in the Cython source files at pandas/_libs/window/, implementing proper handling for subnormal numbers in the rolling aggregation functions.