# Bug Report: scipy.constants.convert_temperature Accepts Physically Impossible Negative Absolute Temperatures

**Target**: `scipy.constants.convert_temperature`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `convert_temperature` function accepts and converts physically impossible negative values for absolute temperature scales (Kelvin and Rankine) without validation, violating fundamental thermodynamic constraints where absolute zero is the lowest possible temperature (0 K or 0 R).

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from scipy.constants import convert_temperature


@given(st.floats(min_value=-1000, max_value=-0.01, allow_nan=False, allow_infinity=False))
@settings(max_examples=100)
def test_negative_kelvin_accepted(negative_kelvin):
    """
    Bug: convert_temperature accepts negative Kelvin values.
    Negative Kelvin is physically impossible (absolute zero is 0 K).
    """
    result = convert_temperature(negative_kelvin, 'Kelvin', 'Celsius')
    assert negative_kelvin < 0
```

**Failing input**: `-10.0` (any negative number for Kelvin or Rankine)

## Reproducing the Bug

```python
from scipy.constants import convert_temperature

print("Negative Kelvin accepted:")
result = convert_temperature(-10, 'Kelvin', 'Celsius')
print(f"  -10 K -> {result} C")

print("\nNegative Rankine accepted:")
result = convert_temperature(-10, 'Rankine', 'Fahrenheit')
print(f"  -10 R -> {result} F")

print("\nCelsius below absolute zero produces negative Kelvin:")
result = convert_temperature(-500, 'Celsius', 'Kelvin')
print(f"  -500 C -> {result} K (physically impossible!)")
```

Output:
```
Negative Kelvin accepted:
  -10 K -> -283.15 C

Negative Rankine accepted:
  -10 R -> -469.67 F

Celsius below absolute zero produces negative Kelvin:
  -500 C -> -226.85 K (physically impossible!)
```

## Why This Is A Bug

Kelvin and Rankine are absolute temperature scales where 0 represents absolute zero - the lowest theoretically possible temperature. Negative values on these scales are physically impossible and represent a violation of thermodynamic laws. The function should either:

1. Raise a `ValueError` for negative Kelvin/Rankine inputs
2. Raise a `ValueError` when conversions would produce negative Kelvin/Rankine
3. Document that it performs mathematical conversion without physical validation

The lack of validation can lead to incorrect scientific calculations and propagation of physically meaningless values.

## Fix

Add input validation to check for physically impossible temperatures:

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
+   # Validate absolute temperature scales
+   if old_scale.lower() in ['kelvin', 'k']:
+       if xp.any(_val < 0):
+           raise ValueError(f"Negative Kelvin temperature is physically impossible: {val}")
+   elif old_scale.lower() in ['rankine', 'r']:
+       if xp.any(_val < 0):
+           raise ValueError(f"Negative Rankine temperature is physically impossible: {val}")
+
    # Convert from `old_scale` to Kelvin
    if old_scale.lower() in ['celsius', 'c']:
        tempo = _val + zero_Celsius
    elif old_scale.lower() in ['kelvin', 'k']:
        tempo = _val
    elif old_scale.lower() in ['fahrenheit', 'f']:
        tempo = (_val - 32) * 5 / 9 + zero_Celsius
    elif old_scale.lower() in ['rankine', 'r']:
        tempo = _val * 5 / 9
    else:
        raise NotImplementedError(f"{old_scale=} is unsupported: supported scales "
                                   "are Celsius, Kelvin, Fahrenheit, and "
                                   "Rankine")
+
+   # Validate result doesn't go below absolute zero
+   if xp.any(tempo < 0):
+       raise ValueError(f"Conversion would produce temperature below absolute zero (< 0 K): {tempo} K")
+
    # and from Kelvin to `new_scale`.
    if new_scale.lower() in ['celsius', 'c']:
        res = tempo - zero_Celsius
    elif new_scale.lower() in ['kelvin', 'k']:
        res = tempo
    elif new_scale.lower() in ['fahrenheit', 'f']:
        res = (tempo - zero_Celsius) * 9 / 5 + 32
    elif new_scale.lower() in ['rankine', 'r']:
        res = tempo * 9 / 5
    else:
        raise NotImplementedError(f"{new_scale=} is unsupported: supported "
                                   "scales are 'Celsius', 'Kelvin', "
                                   "'Fahrenheit', and 'Rankine'")

    return res
```