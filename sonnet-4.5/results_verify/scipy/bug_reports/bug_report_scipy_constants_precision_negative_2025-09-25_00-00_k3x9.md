# Bug Report: scipy.constants.precision Returns Negative Values

**Target**: `scipy.constants.precision`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `precision()` function in `scipy.constants` returns negative values for physical constants that have negative values, which is physically nonsensical. Precision (relative uncertainty) should always be non-negative.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from scipy.constants import precision, find


@given(st.sampled_from(find()))
def test_precision_is_always_nonnegative(key):
    prec = precision(key)
    assert prec >= 0, f"precision('{key}') returned {prec}, which is negative"
```

**Failing input**: `'neutron to shielded proton mag. mom. ratio'`

## Reproducing the Bug

```python
from scipy.constants import precision, physical_constants

key = 'neutron to shielded proton mag. mom. ratio'
prec = precision(key)
print(f"precision('{key}') = {prec}")

value, unit, uncertainty = physical_constants[key]
print(f"Value: {value}")
print(f"Uncertainty: {uncertainty}")
print(f"Calculated precision: {uncertainty / value}")
print(f"Expected (non-negative): {abs(uncertainty / value)}")
```

Output:
```
precision('neutron to shielded proton mag. mom. ratio') = -2.3357768576309262e-07
Value: -0.68499694
Uncertainty: 1.6e-07
Calculated precision: -2.3357768576309262e-07
Expected (non-negative): 2.3357768576309262e-07
```

## Why This Is A Bug

The `precision()` function computes relative precision as `uncertainty / value`. For physical constants with negative values (such as magnetic moment ratios), this calculation produces negative results. However, precision represents relative uncertainty and should always be non-negative, regardless of the sign of the measured value.

This affects 33 physical constants in the CODATA database, including:
- 'neutron to shielded proton mag. mom. ratio'
- 'deuteron-electron mag. mom. ratio'
- 'electron mag. mom.'
- 'electron g factor'
- And 29 others

## Fix

```diff
--- a/scipy/constants/_codata.py
+++ b/scipy/constants/_codata.py
@@ -1,7 +1,7 @@
 def precision(key: str) -> float:
     """
     Relative precision in physical_constants indexed by key
     ...
     """
     _check_obsolete(key)
-    return physical_constants[key][2] / physical_constants[key][0]
+    return abs(physical_constants[key][2] / physical_constants[key][0])
```