# Bug Report: numpy.mean Violates Min-Max Bounds

**Target**: `numpy.mean`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.mean` can return values outside the range `[min, max]` of the input array due to floating-point accumulation errors, violating a fundamental statistical property.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays


@given(
    arrays(
        dtype=np.float64,
        shape=st.integers(1, 100),
        elements=st.floats(allow_nan=False, allow_infinity=False, min_value=-1000, max_value=1000)
    )
)
@settings(max_examples=2000)
def test_mean_respects_bounds(arr):
    mean_val = np.mean(arr)
    min_val = np.min(arr)
    max_val = np.max(arr)
    assert min_val <= mean_val <= max_val
```

**Failing input**: Array of 29 identical values `1.46875144e-290`

## Reproducing the Bug

```python
import numpy as np

arr = np.array([1.46875144e-290] * 29)

mean_val = np.mean(arr)
min_val = np.min(arr)
max_val = np.max(arr)

print(f"Min:  {min_val:.20e}")
print(f"Mean: {mean_val:.20e}")
print(f"Max:  {max_val:.20e}")
print(f"min <= mean <= max: {min_val <= mean_val <= max_val}")
```

Output:
```
Min:  1.46875143668299264372e-290
Mean: 1.46875143668299288014e-290
Max:  1.46875143668299264372e-290
min <= mean <= max: False
```

## Why This Is A Bug

The mathematical property `min(X) <= mean(X) <= max(X)` is a fundamental statistical invariant that should always hold for any dataset. When all array elements are identical, the mean should exactly equal that value, but floating-point accumulation in `sum(arr) / len(arr)` introduces rounding errors.

While the documentation mentions potential inaccuracy for floating-point inputs, it does not warn that basic statistical properties like bounds preservation can be violated. This behavior is surprising and could cause downstream issues in statistical computations.

## Fix

NumPy could use a more numerically stable summation algorithm for `mean()`, such as pairwise summation or Kahan summation. Testing shows that Kahan summation avoids this issue:

```python
arr = np.array([value] * n)

mean_naive = np.sum(arr) / len(arr)

c = 0.0
s = 0.0
for x in arr:
    y = x - c
    t = s + y
    c = (t - s) - y
    s = t
mean_kahan = s / len(arr)

print(f"Naive mean violates bounds: {mean_naive > np.max(arr)}")
print(f"Kahan mean respects bounds: {np.min(arr) <= mean_kahan <= np.max(arr)}")
```

Alternatively, document explicitly that `min <= mean <= max` may not hold for float64 arrays, though this would be surprising behavior for a statistical function.