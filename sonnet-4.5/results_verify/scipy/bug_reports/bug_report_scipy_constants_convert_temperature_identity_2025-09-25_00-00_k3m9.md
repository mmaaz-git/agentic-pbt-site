# Bug Report: scipy.constants.convert_temperature Identity Conversion Loss

**Target**: `scipy.constants.convert_temperature`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When `convert_temperature` is called with the same scale for both `old_scale` and `new_scale` (identity conversion), it loses precision for small temperature values due to catastrophic cancellation in floating-point arithmetic. For very small values (< 1e-20), the result becomes 0.0 instead of returning the original value.

## Property-Based Test

```python
import math
from hypothesis import given, strategies as st
from scipy.constants import convert_temperature


@given(st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False))
def test_same_scale_returns_same_value(temp):
    result = convert_temperature(temp, 'Celsius', 'Celsius')
    assert math.isclose(result, temp, rel_tol=1e-10)
```

**Failing input**: `temp=1.1754943508222875e-38`

## Reproducing the Bug

```python
from scipy.constants import convert_temperature

temp = 1e-10
result = convert_temperature(temp, 'Celsius', 'Celsius')
print(f"Input: {temp}, Result: {result}, Expected: {temp}")

temp2 = 1e-20
result2 = convert_temperature(temp2, 'Celsius', 'Celsius')
print(f"Input: {temp2}, Result: {result2}, Expected: {temp2}")
```

Output:
```
Input: 1e-10, Result: 9.998757e-11, Expected: 1e-10
Input: 1e-20, Result: 0.0, Expected: 1e-20
```

## Why This Is A Bug

The function performs identity conversion by first converting to Kelvin and then back to the original scale. For Celsius to Celsius, this means:
1. `tempo = val + 273.15` (Celsius to Kelvin)
2. `res = tempo - 273.15` (Kelvin back to Celsius)

When `val` is very small (e.g., 1e-38), the addition `val + 273.15` loses the small value due to floating-point precision limits. The subsequent subtraction cannot recover it, resulting in precision loss or complete data loss (0.0).

This violates the mathematical identity property: `f(x, A, A) = x` for all valid `x`.

## Fix

Add an early return when `old_scale == new_scale` to avoid the unnecessary conversion through Kelvin:

```diff
@xp_capabilities()
def convert_temperature(
    val: "npt.ArrayLike",
    old_scale: str,
    new_scale: str,
) -> Any:
    xp = array_namespace(val)
    _val = _asarray(val, xp=xp, subok=True)
+
+   # Early return for identity conversion to avoid precision loss
+   if old_scale.lower() == new_scale.lower():
+       return _val
+
    # Convert from `old_scale` to Kelvin
    if old_scale.lower() in ['celsius', 'c']:
        tempo = _val + zero_Celsius
```

Alternative fix: normalize scale names and compare normalized versions to handle cases like 'Celsius' vs 'C':

```diff
@xp_capabilities()
def convert_temperature(
    val: "npt.ArrayLike",
    old_scale: str,
    new_scale: str,
) -> Any:
    xp = array_namespace(val)
    _val = _asarray(val, xp=xp, subok=True)
+
+   # Normalize scale names for comparison
+   def normalize_scale(scale):
+       scale_lower = scale.lower()
+       if scale_lower in ['celsius', 'c']:
+           return 'celsius'
+       elif scale_lower in ['kelvin', 'k']:
+           return 'kelvin'
+       elif scale_lower in ['fahrenheit', 'f']:
+           return 'fahrenheit'
+       elif scale_lower in ['rankine', 'r']:
+           return 'rankine'
+       return scale_lower
+
+   # Early return for identity conversion to avoid precision loss
+   if normalize_scale(old_scale) == normalize_scale(new_scale):
+       return _val
+
    # Convert from `old_scale` to Kelvin
    if old_scale.lower() in ['celsius', 'c']:
        tempo = _val + zero_Celsius
```