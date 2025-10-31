# Bug Report: scipy.constants.precision() Returns Negative Values

**Target**: `scipy.constants.precision()`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `scipy.constants.precision()` function returns negative values for physical constants that have negative values. This violates standard metrological convention where relative precision/uncertainty should always be non-negative.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from scipy.constants import find, precision

def test_precision_is_non_negative():
    all_keys = find(None, disp=False)
    for key in all_keys:
        prec = precision(key)
        assert prec >= 0, f"precision('{key}') = {prec}, should be non-negative"
```

**Failing input**: Any physical constant with a negative value, e.g., `'Sackur-Tetrode constant (1 K, 100 kPa)'`

## Reproducing the Bug

```python
from scipy.constants import precision, value, physical_constants

key = 'Sackur-Tetrode constant (1 K, 100 kPa)'
constant_value = value(key)
constant_precision = precision(key)

print(f"Value: {constant_value}")
print(f"Precision: {constant_precision}")

raw = physical_constants[key]
print(f"Raw data: {raw}")
```

**Output:**
```
Value: -1.15170753496
Precision: -4.080897152559861e-10
Raw data: (-1.15170753496, '', 4.7e-10)
```

The precision is negative because the function computes `uncertainty / value`, and when the value is negative, the result is negative. There are 58 such constants affected.

## Why This Is A Bug

1. **Violates standard metrological definitions**: In metrology, relative uncertainty (precision) is defined as `|uncertainty / value|`, which is always non-negative. The magnitude represents the fractional uncertainty.

2. **Misleading function name**: "Precision" suggests a magnitude, not a signed value. The sign of a constant's value is unrelated to its precision.

3. **Inconsistent with docstring example**: The docstring example shows `constants.precision('proton mass')` returning `5.1e-37`, a positive value, setting the expectation that precision is non-negative.

4. **Unexpected for users**: Users would reasonably expect precision to be non-negative when using it in calculations or comparisons.

## Fix

```diff
--- a/scipy/constants/_codata.py
+++ b/scipy/constants/_codata.py
@@ -85,7 +85,7 @@ def precision(key: str) -> float:
     >>> constants.precision('proton mass')
     5.1e-37

     """
     _check_obsolete(key)
-    return physical_constants[key][2] / physical_constants[key][0]
+    return abs(physical_constants[key][2] / physical_constants[key][0])
```

This simple fix ensures that precision always returns a non-negative value, matching standard metrological definitions and user expectations.