# Bug Report: numpy.random.negative_binomial Returns INT64_MIN for p=0

**Target**: `numpy.random.negative_binomial`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When `p=0` is passed to `numpy.random.negative_binomial`, it returns INT64_MIN (-9223372036854775808) instead of raising a ValueError or returning a mathematically meaningful value.

## Property-Based Test

```python
import numpy as np
import numpy.random
from hypothesis import given, settings, strategies as st


@given(st.integers(min_value=1, max_value=100))
@settings(max_examples=500)
def test_negative_binomial_p_zero_returns_garbage(n):
    result = numpy.random.negative_binomial(n, p=0, size=10)

    assert np.all(result == np.iinfo(np.int64).min), \
        f"negative_binomial(n={n}, p=0) returns INT64_MIN instead of raising error or valid value"
```

**Failing input**: `n=1, p=0, size=10`

## Reproducing the Bug

```python
import numpy as np
import numpy.random

numpy.random.seed(42)
result = numpy.random.negative_binomial(1, 0, size=5)
print(result)

expected_min_int64 = np.iinfo(np.int64).min
assert np.all(result == expected_min_int64)
```

## Why This Is A Bug

The documentation states that `p` should be in the interval `[0, 1]`, explicitly allowing `p=0`. However, when `p=0` is passed:

1. The function returns `-9223372036854775808` (INT64_MIN), which is clearly garbage
2. This violates the documented return type: "number of failures before n successes"
3. The newer `numpy.random.Generator.negative_binomial` API correctly raises `ValueError: p <= 0` for this case
4. When `p=1`, the function correctly returns 0, showing it can handle edge cases properly

The mathematically correct behavior when `p=0` (probability of success is 0) would be either:
- Raise a `ValueError` (preferred, matches new Generator API)
- Return infinity (mathematically correct but impractical)

## Fix

The legacy `RandomState.negative_binomial` should validate that `p > 0` (or at minimum `p > 0.0`), matching the behavior of the newer `Generator.negative_binomial` implementation. The validation likely needs to be added in the Cython code at `numpy/random/mtrand.pyx` around line 3604 where the parameter validation occurs.