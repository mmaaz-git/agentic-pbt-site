# Bug Report: numpy.random.uniform Violates Upper Bound Exclusion

**Target**: `numpy.random.uniform`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`numpy.random.uniform(low, high)` is documented to return values in the half-open interval `[low, high)` (excludes high), but violates this contract by returning exactly `high` when the range is very small (on the order of 5e-324).

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st, assume
import numpy.random as nr


@given(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10),
       st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10))
@settings(max_examples=1000)
def test_uniform_bounds(low, high):
    assume(low < high)
    result = nr.uniform(low, high)
    assert low <= result < high, f"uniform({low}, {high}) = {result} not in range"
```

**Failing input**: `low=-5e-324, high=0.0`

## Reproducing the Bug

```python
import numpy.random as nr

nr.seed(42)
result = nr.uniform(-5e-324, 0.0)
print(f"Result: {result}")
print(f"Violates upper bound: {result >= 0.0}")

violations = 0
nr.seed(123)
for i in range(10000):
    result = nr.uniform(-5e-324, 0.0)
    if result >= 0.0:
        violations += 1

print(f"Violations: {violations}/10000 ({100*violations/10000:.1f}%)")
```

Output:
```
Result: 0.0
Violates upper bound: True
Violations: 4924/10000 (49.2%)
```

## Why This Is A Bug

The documentation explicitly states: "Samples are uniformly distributed over the half-open interval `[low, high)` (includes low, but excludes high)."

When `low=-5e-324` and `high=0.0`, approximately 50% of samples return exactly `0.0`, violating the documented contract that `high` should be excluded. This happens consistently with tiny ranges near floating-point precision limits.

While this is an extreme edge case, the function makes no mention of range size limitations, and violating the documented API contract is a bug regardless of practical impact.

## Fix

This appears to be a floating-point precision issue in the implementation. The algorithm likely uses `low + (high - low) * random_unit` where `random_unit` is in `[0, 1)`. However, when `high - low` is extremely small, the multiplication and addition can produce exactly `high` due to rounding.

A potential fix would be to add a check that clamps the result to be strictly less than `high`:

```python
result = low + (high - low) * random_unit
if result >= high:
    result = np.nextafter(high, low)
```

However, the actual implementation is in C, so the fix would need to be applied there.