# Bug Report: pandas.core.window.rolling Returns Negative Variance and Inconsistent with Std

**Target**: `pandas.core.window.rolling.Rolling.var` and `pandas.core.window.rolling.Rolling.std`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `Rolling.var()` method returns mathematically impossible negative variance values and produces results inconsistent with `Rolling.std()` squared when processing data with large magnitude differences.

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

<details>

<summary>
**Failing input**: `values=[32769.0, 1e-09, 0.0], window=2`
</summary>
```
Running property-based test for pandas rolling variance...
Testing that rolling.var() == rolling.std() ** 2
============================================================
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/30
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_rolling_variance_equals_std_squared FAILED                 [100%]

=================================== FAILURES ===================================
___________________ test_rolling_variance_equals_std_squared ___________________
hypo.py:6: in test_rolling_variance_equals_std_squared
    values=st.lists(
               ^^^
hypo.py:26: in test_rolling_variance_equals_std_squared
    assert math.isclose(rolling_var.iloc[i], expected_var, rel_tol=1e-9, abs_tol=1e-9), \
E   AssertionError: At index 2: var -1.1920928955028125e-07 != std^2 0.0
E   assert False
E    +  where False = <built-in function isclose>(np.float64(-1.1920928955028125e-07), np.float64(0.0), rel_tol=1e-09, abs_tol=1e-09)
E    +    where <built-in function isclose> = math.isclose
E   Falsifying example: test_rolling_variance_equals_std_squared(
E       values=[32769.0, 1e-09, 0.0],
E       window=2,
E   )
=============================== warnings summary ===============================
../../../../miniconda/lib/python3.13/site-packages/_pytest/config/__init__.py:1290
  /home/npc/miniconda/lib/python3.13/site-packages/_pytest/config/__init__.py:1290: PytestAssertRewriteWarning: Module already imported so cannot be rewritten; _hypothesis_globals
    self._mark_plugins_for_rewrite(hook, disable_autoload)

../../../../miniconda/lib/python3.13/site-packages/_pytest/config/__init__.py:1290
  /home/npc/miniconda/lib/python3.13/site-packages/_pytest/config/__init__.py:1290: PytestAssertRewriteWarning: Module already imported so cannot be rewritten; hypothesis
    self._mark_plugins_for_rewrite(hook, disable_autoload)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED hypo.py::test_rolling_variance_equals_std_squared - AssertionError: At...
======================== 1 failed, 2 warnings in 0.03s =========================
```
</details>

## Reproducing the Bug

```python
import pandas as pd
import numpy as np

print("Demonstrating pandas rolling variance bug")
print("=" * 60)

# The exact failing case from Hypothesis test
print("\nFailing case from Hypothesis: values=[32769.0, 1e-09, 0.0], window=2")
print("-" * 60)
values = [32769.0, 1e-09, 0.0]
s = pd.Series(values)
rolling_var = s.rolling(window=2).var()
rolling_std = s.rolling(window=2).std()

print("Series values:", values)
print("\nRolling variance:", rolling_var.values)
print("Rolling std^2:   ", (rolling_std ** 2).values)

print("\nAt index 2 (window: [1e-09, 0.0]):")
print(f"  Pandas rolling variance: {rolling_var.iloc[2]}")
print(f"  Pandas rolling std^2:    {rolling_std.iloc[2] ** 2}")
print(f"  NumPy variance (ddof=1): {np.var([1e-09, 0.0], ddof=1)}")

# Check mathematical violations
print("\nMathematical Violations:")
if rolling_var.iloc[2] < 0:
    print(f"  ✗ Variance is NEGATIVE: {rolling_var.iloc[2]}")
    print(f"    (Variance must always be non-negative)")

if abs(rolling_var.iloc[2] - rolling_std.iloc[2] ** 2) > 1e-10:
    print(f"  ✗ var != std^2:")
    print(f"    var = {rolling_var.iloc[2]}")
    print(f"    std^2 = {rolling_std.iloc[2] ** 2}")
    print(f"    (By definition, variance must equal standard deviation squared)")

# Additional test cases showing the pattern
print("\n" + "=" * 60)
print("Additional test cases showing the pattern:")
print("=" * 60)

test_cases = [
    [131073.0, 1e-05, 0.0],
    [1e10, 1e-10, 0.0],
    [1e8, -1e-8, 0.0]
]

for values in test_cases:
    s = pd.Series(values)
    rolling_var = s.rolling(window=2).var()
    rolling_std = s.rolling(window=2).std()

    print(f"\nValues: {values}")

    # Check last window (indices 1 and 2)
    last_idx = len(values) - 1
    var_val = rolling_var.iloc[last_idx]
    std_squared = rolling_std.iloc[last_idx] ** 2

    print(f"  Window [{values[-2]}, {values[-1]}]:")
    print(f"    Rolling variance: {var_val:20.15e}")
    print(f"    Rolling std^2:   {std_squared:20.15e}")

    if var_val < 0:
        print(f"    ✗ NEGATIVE variance!")
    if abs(var_val - std_squared) > 1e-10:
        print(f"    ✗ var != std^2 (difference: {abs(var_val - std_squared):20.15e})")

print("\n" + "=" * 60)
print("Summary:")
print("-" * 60)
print("pandas.rolling.var() can return:")
print("1. Negative variance values (mathematically impossible)")
print("2. Values inconsistent with rolling.std()^2 (violates definition)")
print("\nThis occurs when windows contain values with large magnitude differences.")
print("The bug is likely due to numerical instability in the incremental variance")
print("calculation algorithm used in pandas' Cython implementation.")
```

<details>

<summary>
Output showing negative variance and inconsistency with std^2
</summary>
```
Demonstrating pandas rolling variance bug
============================================================

Failing case from Hypothesis: values=[32769.0, 1e-09, 0.0], window=2
------------------------------------------------------------
Series values: [32769.0, 1e-09, 0.0]

Rolling variance: [           nan  5.3690368e+08 -1.1920929e-07]
Rolling std^2:    [          nan 5.3690368e+08 0.0000000e+00]

At index 2 (window: [1e-09, 0.0]):
  Pandas rolling variance: -1.1920928955028125e-07
  Pandas rolling std^2:    0.0
  NumPy variance (ddof=1): 5e-19

Mathematical Violations:
  ✗ Variance is NEGATIVE: -1.1920928955028125e-07
    (Variance must always be non-negative)
  ✗ var != std^2:
    var = -1.1920928955028125e-07
    std^2 = 0.0
    (By definition, variance must equal standard deviation squared)

============================================================
Additional test cases showing the pattern:
============================================================

Values: [131073.0, 1e-05, 0.0]
  Window [1e-05, 0.0]:
    Rolling variance: -1.907298632812500e-06
    Rolling std^2:   0.000000000000000e+00
    ✗ NEGATIVE variance!
    ✗ var != std^2 (difference: 1.907298632812500e-06)

Values: [10000000000.0, 1e-10, 0.0]
  Window [1e-10, 0.0]:
    Rolling variance: 5.000000000000000e-21
    Rolling std^2:   5.000000000000000e-21

Values: [100000000.0, -1e-08, 0.0]
  Window [-1e-08, 0.0]:
    Rolling variance: 5.000000000000001e-17
    Rolling std^2:   5.000000000000001e-17

============================================================
Summary:
------------------------------------------------------------
pandas.rolling.var() can return:
1. Negative variance values (mathematically impossible)
2. Values inconsistent with rolling.std()^2 (violates definition)

This occurs when windows contain values with large magnitude differences.
The bug is likely due to numerical instability in the incremental variance
calculation algorithm used in pandas' Cython implementation.
```
</details>

## Why This Is A Bug

This bug violates two fundamental mathematical axioms:

1. **Variance must be non-negative**: Variance is defined as E[(X - μ)²], the expected value of squared deviations from the mean. Since squaring always produces non-negative values, variance **cannot be negative** by mathematical definition. The pandas documentation states that `var()` computes "sample variance" which must follow this property.

2. **Variance equals standard deviation squared**: By definition, σ² = Var(X) where σ is the standard deviation. The pandas documentation for `Rolling.std()` states it returns "standard deviation" and `Rolling.var()` returns "variance". These two methods must maintain the mathematical relationship var = std², but the bug causes them to return inconsistent values.

The bug manifests specifically when:
- A rolling window processes a large value (e.g., 32769.0)
- Followed by much smaller values in the next window (e.g., [1e-09, 0.0])
- The incremental variance algorithm suffers catastrophic cancellation errors

This results in silent data corruption - no errors or warnings are raised, leading to incorrect statistical analysis in financial modeling, quality control, scientific computing, and any domain relying on rolling statistics.

## Relevant Context

The bug occurs in pandas version 2.3.2 and likely affects earlier versions. The issue stems from the Cython implementation of rolling variance calculations which uses an incremental (online) algorithm for efficiency. When processing values with large magnitude differences, the algorithm accumulates numerical errors due to floating-point precision limitations.

Key implementation files:
- Python interface: `/pandas/core/window/rolling.py` (Rolling.var method)
- Cython implementation: `/pandas/_libs/window/aggregations.pyx` (actual variance calculation)

The rolling variance uses Welford's online algorithm or a similar incremental approach to avoid storing all window values. While computationally efficient, this method is susceptible to catastrophic cancellation when the running sum of squares and the squared running sum have vastly different magnitudes.

Documentation references:
- [pandas.DataFrame.rolling.var](https://pandas.pydata.org/docs/reference/api/pandas.core.window.rolling.Rolling.var.html)
- [pandas.DataFrame.rolling.std](https://pandas.pydata.org/docs/reference/api/pandas.core.window.rolling.Rolling.std.html)

## Proposed Fix

The fix requires modifying the Cython implementation to handle numerical stability. Since the core implementation is in compiled code, here's a high-level approach:

1. **Immediate mitigation**: Add a post-computation check to ensure non-negative variance
2. **Proper fix**: Use a numerically stable variance algorithm

High-level fix approach for the Cython implementation:

```diff
# In pandas/_libs/window/aggregations.pyx (pseudocode)
# In the rolling variance calculation function

def roll_var(...):
    # ... existing calculation code ...

    # Calculate variance using current algorithm
    variance = calculated_variance

+   # Numerical stability check
+   if variance < 0:
+       # For very small negative values due to rounding, clamp to 0
+       if variance > -1e-10:
+           variance = 0.0
+       else:
+           # For larger negative values, recalculate using stable algorithm
+           # Fall back to two-pass calculation for this window
+           variance = stable_variance_calculation(window_values)

+   # Ensure consistency with standard deviation
+   # If std is computed separately, ensure var = std^2
+   if compute_both_var_and_std:
+       std_value = sqrt(max(0, variance))
+       variance = std_value * std_value

    return variance
```

For a more robust solution, consider implementing:
1. Compensated summation (Kahan summation) for the running sums
2. Two-pass algorithm for windows with high dynamic range
3. Pairwise summation to reduce rounding errors

The fix should be applied in the Cython aggregation functions that compute rolling variance, ensuring both mathematical correctness and numerical stability without significantly impacting performance.