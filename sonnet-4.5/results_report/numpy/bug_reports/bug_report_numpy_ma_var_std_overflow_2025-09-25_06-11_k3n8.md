# Bug Report: numpy.ma.var() and std() Auto-Mask Overflow in Unmasked Data

**Target**: `numpy.ma.var()` and `numpy.ma.std()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.ma.var()` and `numpy.ma.std()` return masked scalars (MaskedConstant) when intermediate calculations overflow, even when all input data is unmasked. This is inconsistent with the documented behavior and with regular `numpy.var()`/`numpy.std()` which return inf.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays, array_shapes
import numpy as np
import numpy.ma as ma
import math

@st.composite
def masked_array_strategy(draw, max_dims=1, max_side=20):
    shape = draw(array_shapes(max_dims=max_dims, max_side=max_side))
    data = draw(arrays(dtype=np.float64, shape=shape))
    mask = draw(arrays(dtype=bool, shape=shape))
    return data, mask

@given(masked_array_strategy(max_dims=1, max_side=20))
def test_std_from_unmasked(args):
    data, mask = args
    assume(np.sum(~mask) > 1)

    masked = ma.masked_array(data, mask=mask)
    std_val = ma.std(masked)

    unmasked_data = data[~mask]
    expected_std = np.std(unmasked_data)

    assert math.isclose(std_val, expected_std, rel_tol=1e-10) or (np.isnan(std_val) and np.isnan(expected_std))
```

**Failing input**: `data=[0.0, 1.38204521e+154, 1.38204521e+154, ...] (17 values), mask=[False]*17`

## Reproducing the Bug

```python
import numpy as np
import numpy.ma as ma

data = np.array([0.0, 1.38204521e+154, 1.38204521e+154, 1.38204521e+154,
                 1.38204521e+154, 1.38204521e+154, 1.38204521e+154, 1.38204521e+154,
                 1.38204521e+154, 1.38204521e+154, 1.38204521e+154, 1.38204521e+154,
                 1.38204521e+154, 1.38204521e+154, 1.38204521e+154, 1.38204521e+154,
                 1.38204521e+154])

masked_arr = ma.masked_array(data, mask=[False] * 17)

var_result = ma.var(masked_arr)
std_result = ma.std(masked_arr)

print(f"Input: all unmasked")
print(f"ma.var(): {var_result}, is_masked={ma.is_masked(var_result)}")
print(f"ma.std(): {std_result}, is_masked={ma.is_masked(std_result)}")
print(f"np.var(): {np.var(data)}")
print(f"np.std(): {np.std(data)}")

assert ma.is_masked(var_result)
assert ma.is_masked(std_result)
```

## Why This Is A Bug

1. **Violates documented behavior**: The docstring says "Masked entries are ignored", referring to input masking, not output
2. **Inconsistent with numpy**: `numpy.var()` and `numpy.std()` return inf for the same input
3. **Breaks invariant**: Operations on fully unmasked data should not return masked results
4. **Inconsistent with other operations**: `add()`, `multiply()`, `exp()`, etc. return inf on overflow without masking
5. **Unexpected behavior**: Users expect `var()`/`std()` to return inf (like numpy) when calculations overflow, not a masked value

The root cause is that when computing variance/std, intermediate calculations (squaring deviations from mean) overflow to inf, and this overflow is auto-masked.

## Fix

The fix should make `var()` and `std()` consistent with numpy by returning inf when calculations overflow instead of returning a masked scalar. If auto-masking overflow is desired, it should be:
1. Documented in the docstring
2. Applied consistently to all operations (currently only power/divide/var/std do this)
3. Made configurable

Alternatively, if this masking behavior is intentional, it needs to be clearly documented and applied consistently across all operations.