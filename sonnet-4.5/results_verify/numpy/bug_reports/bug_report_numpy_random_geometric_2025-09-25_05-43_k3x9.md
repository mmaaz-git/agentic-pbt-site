# Bug Report: numpy.random.geometric Integer Overflow with Small Probabilities

**Target**: `numpy.random.geometric`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.random.geometric()` returns the minimum int64 value (-9223372036854775808) instead of valid positive integers when given very small probability values (approximately p < 1e-30), causing silent data corruption.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings, assume


@given(
    st.floats(min_value=1e-100, max_value=1.0, allow_nan=False),
    st.integers(min_value=1, max_value=100)
)
@settings(max_examples=1000)
def test_geometric_always_positive(p, size):
    assume(0 < p <= 1.0)

    result = np.random.geometric(p, size)

    assert np.all(result >= 1), f"geometric({p}) returned invalid values: min={np.min(result)}"
    assert np.issubdtype(result.dtype, np.integer), f"geometric() should return integers"
```

**Failing input**: `p=1e-100`

## Reproducing the Bug

```python
import numpy as np

p = 1e-100
np.random.seed(42)
result = np.random.geometric(p, size=1)[0]

print(f"p = {p}")
print(f"result = {result}")
print(f"Expected: positive integer >= 1")
print(f"Bug confirmed: {result == -9223372036854775808}")
```

Output:
```
p = 1e-100
result = -9223372036854775808
Expected: positive integer >= 1
Bug confirmed: True
```

## Why This Is A Bug

The geometric distribution models the number of Bernoulli trials needed to get one success. By definition, this must be a positive integer (at least 1, since you need at least one trial). The function accepts probability values in the range (0, 1], and while very small probabilities are unusual, they are mathematically valid inputs.

When p is very small (< ~1e-30), the function returns -9223372036854775808 (the minimum value for a signed 64-bit integer), indicating an integer overflow during internal computation. This violates the fundamental mathematical property that geometric distribution values must be >= 1.

This is a HIGH severity bug because:
1. It silently returns incorrect values instead of raising an error
2. It violates documented behavior of the geometric distribution
3. Users may not notice the corruption in large datasets
4. The incorrect value is far from any reasonable result

## Fix

The bug likely occurs in the internal computation when calculating `ceil(log(U) / log(1-p))` for very small p values, causing integer overflow. The fix should either:

1. Validate the input and raise an error for extremely small p values
2. Use extended precision or a different algorithm for small p values
3. Clamp the result to a reasonable maximum value

Without access to the C implementation, a high-level fix would involve adding validation:

```diff
def geometric(p, size=None):
+   if p < 1e-30:
+       raise ValueError("p value too small - may cause integer overflow")
    # ... existing implementation
```

Alternatively, the internal algorithm should be fixed to handle small p values correctly without overflow.