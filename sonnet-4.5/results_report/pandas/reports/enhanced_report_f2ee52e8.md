# Bug Report: pandas.core.window.rolling.Rolling.var Returns Negative Variance Values

**Target**: `pandas.core.window.rolling.Rolling.var`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The pandas rolling variance computation returns mathematically impossible negative values when processing data with extreme numerical ranges, violating the fundamental mathematical property that variance must be non-negative.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, assume, example
import pandas as pd

@given(
    data=st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10), min_size=3, max_size=20),
    window=st.integers(min_value=2, max_value=10)
)
@settings(max_examples=500)
@example(data=[5897791891.464727, -2692142700.7497644, 0.0, 1.0], window=2)
def test_rolling_var_nonnegative(data, window):
    assume(window <= len(data))
    s = pd.Series(data)
    result = s.rolling(window=window).var()
    valid_results = result.dropna()
    for i, val in enumerate(valid_results):
        assert val >= 0, f"Variance should be non-negative at index {i+window-1}, got {val} for data {data}"

# Run the test
if __name__ == "__main__":
    test_rolling_var_nonnegative()
```

<details>

<summary>
**Failing input**: `data=[5897791891.464727, -2692142700.7497644, 0.0, 1.0], window=2`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/27/variance_hypo.py", line 20, in <module>
    test_rolling_var_nonnegative()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/27/variance_hypo.py", line 5, in test_rolling_var_nonnegative
    data=st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10), min_size=3, max_size=20),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
    _raise_to_user(errors, state.settings, [], " in explicit examples")
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/27/variance_hypo.py", line 16, in test_rolling_var_nonnegative
    assert val >= 0, f"Variance should be non-negative at index {i+window-1}, got {val} for data {data}"
           ^^^^^^^^
AssertionError: Variance should be non-negative at index 3, got -8191.5 for data [5897791891.464727, -2692142700.7497644, 0.0, 1.0]
Falsifying explicit example: test_rolling_var_nonnegative(
    data=[5897791891.464727, -2692142700.7497644, 0.0, 1.0],
    window=2,
)
```
</details>

## Reproducing the Bug

```python
import pandas as pd

data = [5897791891.464727, -2692142700.7497644, 0.0, 1.0]
s = pd.Series(data)
result = s.rolling(window=2).var()

print("Rolling variance results:")
print(result)
print()

print(f"At index 3 (window [0.0, 1.0]): {result.iloc[3]}")
print(f"Expected variance for [0.0, 1.0]: 0.5")
print(f"Actual variance: {result.iloc[3]}")
print()

# Verify that this is indeed negative
if result.iloc[3] < 0:
    print(f"ERROR: Variance is negative ({result.iloc[3]}), which is mathematically impossible!")
```

<details>

<summary>
Variance is negative (-8191.5), which is mathematically impossible
</summary>
```
Rolling variance results:
0             NaN
1    3.689349e+19
2    3.623816e+18
3   -8.191500e+03
dtype: float64

At index 3 (window [0.0, 1.0]): -8191.5
Expected variance for [0.0, 1.0]: 0.5
Actual variance: -8191.5

ERROR: Variance is negative (-8191.5), which is mathematically impossible!
```
</details>

## Why This Is A Bug

Variance is mathematically defined as the expected value of the squared deviation from the mean: Var(X) = E[(X - μ)²]. Since it's the average of squared values, variance must always be non-negative (≥ 0). The pandas rolling variance returning -8191.5 for the window [0.0, 1.0] directly contradicts this fundamental mathematical axiom.

The correct variance for the window [0.0, 1.0] should be 0.5, calculated as:
- Mean: (0.0 + 1.0) / 2 = 0.5
- Variance: ((0.0 - 0.5)² + (1.0 - 0.5)²) / 1 = (0.25 + 0.25) / 1 = 0.5

This bug occurs due to accumulated floating-point precision errors when the rolling window processes values of vastly different magnitudes (billions followed by near-zero values). The error accumulates through the incremental variance algorithm, leading to mathematically impossible results.

## Relevant Context

- **GitHub Issue #52407**: The pandas developers have acknowledged this as "clearly a bug" and have welcomed pull requests to fix it
- **Algorithm Used**: pandas uses Welford's online variance algorithm with Kahan summation for numerical stability (located in `pandas/core/_numba/kernels/var_.py` and `pandas/_libs/window/aggregations.pyx`)
- **Affected Versions**: This bug affects the default Cython computation engine. The Numba engine may exhibit similar issues
- **Documentation**: The pandas documentation doesn't explicitly guarantee non-negative variance, but this is a universal mathematical property that users rightfully expect
- **Workarounds**:
  - Normalize data to similar scales before computing variance
  - Use two-pass variance computation for small windows
  - Add post-processing to clamp negative values to zero (though this masks the underlying issue)

## Proposed Fix

The issue requires improving numerical stability in the incremental variance algorithm. Since pandas already uses Welford's algorithm with Kahan summation (state-of-the-art for online variance), the fix should add a fail-safe validation:

```diff
--- a/pandas/core/_numba/kernels/var_.py
+++ b/pandas/core/_numba/kernels/var_.py
@@ -145,7 +145,11 @@ def sliding_var(values, start, end, min_periods, ddof=1):
         else:
             result[i] = np.nan
         else:
-            result[i] = ssqdm_x / (nobs - ddof)
+            variance = ssqdm_x / (nobs - ddof)
+            # Ensure variance is non-negative (handle floating-point errors)
+            if variance < 0 and variance > -1e-10:
+                variance = 0.0
+            result[i] = variance

     return result
```

A more comprehensive fix would involve:
1. Implementing a two-pass algorithm for small windows (< 100 elements) where precision is critical
2. Using extended precision (128-bit floats) for intermediate calculations when available
3. Adding data pre-conditioning when extreme value ranges are detected
4. Documenting known limitations with extreme numerical ranges in the API documentation