# Bug Report: numpy.random.geometric Integer Overflow Returns Negative Values for Small Probabilities

**Target**: `numpy.random.geometric`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.random.geometric()` returns INT64_MIN (-9223372036854775808) instead of positive integers when given probability values smaller than approximately 1e-16, causing silent data corruption that violates the mathematical definition of the geometric distribution.

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


if __name__ == "__main__":
    test_geometric_always_positive()
```

<details>

<summary>
**Failing input**: `p=1e-100, size=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/48/hypo.py", line 20, in <module>
    test_geometric_always_positive()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/48/hypo.py", line 6, in test_geometric_always_positive
    st.floats(min_value=1e-100, max_value=1.0, allow_nan=False),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/48/hypo.py", line 15, in test_geometric_always_positive
    assert np.all(result >= 1), f"geometric({p}) returned invalid values: min={np.min(result)}"
           ~~~~~~^^^^^^^^^^^^^
AssertionError: geometric(1e-100) returned invalid values: min=-9223372036854775808
Falsifying example: test_geometric_always_positive(
    p=1e-100,
    size=1,  # or any other generated value
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py:3186
```
</details>

## Reproducing the Bug

```python
import numpy as np

p = 1e-100
np.random.seed(42)
result = np.random.geometric(p, size=10)

print(f"Testing numpy.random.geometric with p = {p}")
print(f"Generated 10 values: {result}")
print(f"Minimum value: {np.min(result)}")
print(f"Maximum value: {np.max(result)}")
print(f"All values >= 1 (expected): {np.all(result >= 1)}")
print(f"")
print(f"INT64_MIN = -9223372036854775808")
print(f"Bug confirmed (values equal INT64_MIN): {np.all(result == -9223372036854775808)}")
```

<details>

<summary>
All values returned are INT64_MIN (-9223372036854775808) instead of positive integers
</summary>
```
Testing numpy.random.geometric with p = 1e-100
Generated 10 values: [-9223372036854775808 -9223372036854775808 -9223372036854775808
 -9223372036854775808 -9223372036854775808 -9223372036854775808
 -9223372036854775808 -9223372036854775808 -9223372036854775808
 -9223372036854775808]
Minimum value: -9223372036854775808
Maximum value: -9223372036854775808
All values >= 1 (expected): False

INT64_MIN = -9223372036854775808
Bug confirmed (values equal INT64_MIN): True
```
</details>

## Why This Is A Bug

The geometric distribution models the number of Bernoulli trials needed to achieve the first success. By mathematical definition, this must always be a positive integer (at least 1 trial is needed). The NumPy documentation explicitly states that the geometric distribution "is therefore supported on the positive integers, k = 1, 2, ...".

However, when probability values are smaller than approximately 1e-16, the function returns -9223372036854775808 (INT64_MIN), which:

1. **Violates the mathematical definition**: The geometric distribution can only produce positive integers >= 1
2. **Contradicts the documentation**: The docs state the support is "k = 1, 2, ..."
3. **Causes silent data corruption**: No error or warning is raised; the function simply returns invalid negative values
4. **Indicates integer overflow**: The returned value is exactly INT64_MIN, suggesting an overflow in the internal computation

The input values (0 < p <= 1) are within the documented valid range. While probabilities like 1e-100 are small, they are mathematically valid and may occur in scientific computing contexts (e.g., rare event simulation, extreme value theory).

## Relevant Context

Testing reveals the bug threshold is between 1e-16 and 1e-17:
- p = 1e-16: Works correctly (returns positive values)
- p = 1e-17 and smaller: Returns INT64_MIN

The bug likely occurs in the inverse transform method used internally. The geometric distribution is typically sampled using: `ceil(log(U) / log(1-p))` where U is uniform(0,1). For very small p, `log(1-p)` approaches 0, causing the division to overflow.

NumPy documentation: https://numpy.org/doc/stable/reference/random/generated/numpy.random.geometric.html

The function accepts the input without raising any error, implying these probability values should be supported.

## Proposed Fix

Since this involves NumPy's C implementation, a high-level fix would add input validation to prevent silent corruption:

```diff
def geometric(p, size=None):
+   # Prevent integer overflow for extremely small probabilities
+   if np.any(p < 1e-16):
+       raise ValueError(f"Probability p={p} is too small and may cause integer overflow. "
+                       f"Minimum supported value is 1e-16.")
    # ... existing implementation
```

A better fix would handle small probabilities correctly in the C implementation, either by:
1. Using extended precision arithmetic for the logarithm computation
2. Implementing an alternative sampling algorithm for small p values
3. Clamping the result to a reasonable maximum value (e.g., INT64_MAX) with a warning