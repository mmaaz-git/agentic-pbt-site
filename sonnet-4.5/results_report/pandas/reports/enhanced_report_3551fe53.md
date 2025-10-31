# Bug Report: pandas.core.window.expanding.sum Violates Monotonicity When Adding Zero

**Target**: `pandas.core.window.expanding.Expanding.sum`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `expanding().sum()` method in pandas violates the mathematical property of monotonicity when computing cumulative sums of non-negative values. Specifically, adding 0.0 to an existing sum causes the result to decrease by approximately 1.16e-10, which should never happen.

## Property-Based Test

```python
import pandas as pd
from hypothesis import given, strategies as st, settings, Verbosity, example

@given(st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=0, max_value=1e6), min_size=1, max_size=100))
@example([1.023075029544998, 524288.3368640885, 0.0])
@settings(verbosity=Verbosity.verbose, max_examples=100)
def test_expanding_sum_monotonic_for_nonnegative(data):
    s = pd.Series(data)
    result = s.expanding().sum()

    for i in range(1, len(result)):
        if pd.notna(result.iloc[i]) and pd.notna(result.iloc[i-1]):
            assert result.iloc[i] >= result.iloc[i-1], f"Monotonicity violated at position {i}: {result.iloc[i]} < {result.iloc[i-1]}"

# Run the test
if __name__ == "__main__":
    print("Running property-based test to find failures...")
    test_expanding_sum_monotonic_for_nonnegative()
```

<details>

<summary>
**Failing input**: `[1.023075029544998, 524288.3368640885, 0.0]`
</summary>
```
Running property-based test to find failures...
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/29/hypo.py", line 18, in <module>
    test_expanding_sum_monotonic_for_nonnegative()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/29/hypo.py", line 5, in test_expanding_sum_monotonic_for_nonnegative
    @example([1.023075029544998, 524288.3368640885, 0.0])
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
    _raise_to_user(errors, state.settings, [], " in explicit examples")
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/29/hypo.py", line 13, in test_expanding_sum_monotonic_for_nonnegative
    assert result.iloc[i] >= result.iloc[i-1], f"Monotonicity violated at position {i}: {result.iloc[i]} < {result.iloc[i-1]}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Monotonicity violated at position 2: 524289.359939118 < 524289.3599391181
Falsifying explicit example: test_expanding_sum_monotonic_for_nonnegative(
    data=[1.023075029544998, 524288.3368640885, 0.0],
)
```
</details>

## Reproducing the Bug

```python
import pandas as pd
import numpy as np

data = [1.023075029544998, 524288.3368640885, 0.0]
s = pd.Series(data)
result = s.expanding().sum()

print(f"Input data: {data}")
print(f"Expanding sum results:")
for i in range(len(result)):
    print(f"  Position {i}: {result.iloc[i]:.20f}")

print(f"\nComparison:")
print(f"Position 1 sum: {result.iloc[1]:.20f}")
print(f"Position 2 sum: {result.iloc[2]:.20f}")
print(f"Difference: {result.iloc[1] - result.iloc[2]:.20e}")

# Check monotonicity
print(f"\nMonotonicity check:")
print(f"result[2] >= result[1]: {result.iloc[2] >= result.iloc[1]}")

# Compare with numpy cumsum
numpy_result = np.cumsum(data)
print(f"\nNumPy cumsum results:")
for i in range(len(numpy_result)):
    print(f"  Position {i}: {numpy_result[i]:.20f}")

# Compare with Python sum
print(f"\nPython sum results:")
for i in range(len(data)):
    python_sum = sum(data[:i+1])
    print(f"  Position {i}: {python_sum:.20f}")

assert result.iloc[2] >= result.iloc[1], "Monotonicity violated!"
```

<details>

<summary>
AssertionError: Monotonicity violated!
</summary>
```
Input data: [1.023075029544998, 524288.3368640885, 0.0]
Expanding sum results:
  Position 0: 1.02307502954499796033
  Position 1: 524289.35993911814875900745
  Position 2: 524289.35993911803234368563

Comparison:
Position 1 sum: 524289.35993911814875900745
Position 2 sum: 524289.35993911803234368563
Difference: 1.16415321826934814453e-10

Monotonicity check:
result[2] >= result[1]: False

NumPy cumsum results:
  Position 0: 1.02307502954499796033
  Position 1: 524289.35993911814875900745
  Position 2: 524289.35993911814875900745

Python sum results:
  Position 0: 1.02307502954499796033
  Position 1: 524289.35993911814875900745
  Position 2: 524289.35993911814875900745
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/29/repo.py", line 34, in <module>
    assert result.iloc[2] >= result.iloc[1], "Monotonicity violated!"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Monotonicity violated!
```
</details>

## Why This Is A Bug

This behavior violates the fundamental mathematical property that cumulative sums of non-negative values must be monotonically non-decreasing. When we add 0.0 to an existing sum of 524289.35993911814875900745, the result decreases to 524289.35993911803234368563 - a loss of approximately 1.16e-10.

This contradicts basic mathematics where sum(a, b, 0) must equal sum(a, b). The issue is particularly concerning because:

1. **Mathematical Invariant Violation**: For any non-negative sequence, the cumulative sum at position i must be greater than or equal to the sum at position i-1. This is a fundamental property that many algorithms depend on.

2. **Inconsistency with Standard Implementations**: Both NumPy's `cumsum` and Python's built-in `sum` correctly maintain the value at 524289.35993911814875900745 when adding 0.0, while pandas produces a different (and smaller) result.

3. **Precision Loss Beyond Floating-Point Tolerance**: While floating-point arithmetic inherently has precision limits, the directional error (decrease when adding a non-negative value) indicates a bug in the accumulation algorithm rather than acceptable rounding error.

4. **Potential Kahan Summation Failure**: According to pandas GitHub issue #13254, the expanding sum should use Kahan summation for numerical stability. If implemented correctly, Kahan summation should prevent exactly this type of accumulated error. The presence of this bug suggests the implementation may be faulty.

## Relevant Context

The pandas documentation for `expanding().sum()` mentions that "Certain Scipy window types require additional parameters to be passed in the aggregation function" and warns about numerical imprecision in the Notes section. However, this warning does not excuse violations of fundamental mathematical properties like monotonicity.

The underlying implementation appears to be in the Cython code at `pandas._libs.window.aggregations.roll_sum`, which is supposed to implement Kahan summation for numerical stability. The fact that this error occurs suggests either:
- Kahan summation is not properly implemented
- There's a bug in how the cumulative state is maintained
- The algorithm switches between different summation methods incorrectly

Pandas version tested: 2.3.2

## Proposed Fix

The issue likely resides in the Cython implementation of the rolling sum algorithm. A proper fix would require implementing or fixing the Kahan summation algorithm to maintain numerical precision. Here's a high-level overview of what should be implemented:

The Kahan summation algorithm should maintain a compensation term to track lost precision:

```python
def kahan_sum(values):
    sum = 0.0
    c = 0.0  # compensation for lost low-order bits
    for value in values:
        y = value - c  # compensate for error
        t = sum + y    # new sum
        c = (t - sum) - y  # new compensation
        sum = t
    return sum
```

For the expanding sum, this needs to be adapted to maintain state across the expanding window. The current implementation appears to lose precision when transitioning between window positions, particularly when adding values that don't change the sum (like 0.0).

Without access to the exact Cython implementation, a specific patch cannot be provided, but the fix would involve ensuring that:
1. The compensation term is properly maintained across all window positions
2. The algorithm doesn't reset or lose precision when adding zero values
3. The implementation matches the numerical stability of NumPy's cumsum