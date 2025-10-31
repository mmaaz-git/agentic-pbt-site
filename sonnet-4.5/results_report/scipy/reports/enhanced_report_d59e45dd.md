# Bug Report: scipy.constants.precision() Returns Negative Values for Negative-Valued Constants

**Target**: `scipy.constants.precision()`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `scipy.constants.precision()` function returns negative values when called on physical constants that have negative values, violating standard metrological conventions where relative precision/uncertainty should always be non-negative.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Hypothesis test for scipy.constants.precision() non-negativity property."""

from hypothesis import given, strategies as st, settings
from scipy.constants import find, precision

def test_precision_is_non_negative():
    """Test that precision() returns non-negative values for all physical constants."""
    all_keys = find(None, disp=False)
    failures = []

    for key in all_keys:
        prec = precision(key)
        if prec < 0:
            failures.append((key, prec))

    if failures:
        print(f"Found {len(failures)} constants with negative precision:")
        for key, prec in failures[:5]:  # Show first 5 failures
            print(f"  precision('{key}') = {prec}")
        if len(failures) > 5:
            print(f"  ... and {len(failures) - 5} more")

        # Raise assertion with the first failing example
        key, prec = failures[0]
        assert prec >= 0, f"precision('{key}') = {prec}, should be non-negative"
    else:
        print("All precision values are non-negative (test passed)")

if __name__ == "__main__":
    test_precision_is_non_negative()
```

<details>

<summary>
**Failing input**: `'Sackur-Tetrode constant (1 K, 100 kPa)'`
</summary>
```
Found 33 constants with negative precision:
  precision('Sackur-Tetrode constant (1 K, 100 kPa)') = -4.080897152559861e-10
  precision('Sackur-Tetrode constant (1 K, 101.325 kPa)') = -4.034783191172332e-10
  precision('deuteron-electron mag. mom. ratio') = -2.572708190541329e-09
  precision('deuteron-neutron mag. mom. ratio') = -2.4542257885940616e-07
  precision('electron charge to mass quotient') = -3.1270965612142976e-10
  ... and 28 more
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/46/hypo.py", line 31, in <module>
    test_precision_is_non_negative()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/46/hypo.py", line 26, in test_precision_is_non_negative
    assert prec >= 0, f"precision('{key}') = {prec}, should be non-negative"
           ^^^^^^^^^
AssertionError: precision('Sackur-Tetrode constant (1 K, 100 kPa)') = -4.080897152559861e-10, should be non-negative
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction case for scipy.constants.precision() returning negative values."""

from scipy.constants import precision, value, physical_constants

# Test with the Sackur-Tetrode constant which has a negative value
key = 'Sackur-Tetrode constant (1 K, 100 kPa)'

# Get the value and precision
constant_value = value(key)
constant_precision = precision(key)

print(f"Constant: {key}")
print(f"Value: {constant_value}")
print(f"Precision: {constant_precision}")
print(f"Is precision negative? {constant_precision < 0}")

# Show the raw data from physical_constants
raw = physical_constants[key]
print(f"\nRaw data from physical_constants:")
print(f"  Value: {raw[0]}")
print(f"  Unit: {raw[1]}")
print(f"  Uncertainty: {raw[2]}")

# Demonstrate the calculation
calculated_precision = raw[2] / raw[0]
print(f"\nCalculated precision (uncertainty/value): {calculated_precision}")
print(f"This matches precision() output: {calculated_precision == constant_precision}")

# What it should be according to metrological standards
correct_precision = abs(raw[2] / raw[0])
print(f"\nCorrect precision (|uncertainty/value|): {correct_precision}")
print(f"This would be non-negative: {correct_precision >= 0}")
```

<details>

<summary>
Precision returns negative for negative-valued constant
</summary>
```
Constant: Sackur-Tetrode constant (1 K, 100 kPa)
Value: -1.15170753496
Precision: -4.080897152559861e-10
Is precision negative? True

Raw data from physical_constants:
  Value: -1.15170753496
  Unit:
  Uncertainty: 4.7e-10

Calculated precision (uncertainty/value): -4.080897152559861e-10
This matches precision() output: True

Correct precision (|uncertainty/value|): 4.080897152559861e-10
This would be non-negative: True
```
</details>

## Why This Is A Bug

This violates expected behavior in three critical ways:

1. **Violates International Metrological Standards**: According to NIST, CODATA, and standard metrology practice, relative uncertainty (precision) is defined as ur(y) = u(y)/|y| where the denominator uses the absolute value. This ensures relative precision is always non-negative, as it represents a magnitude of uncertainty, not a signed value.

2. **Contradicts Function Semantics**: The function is named "precision", which inherently implies a magnitude or measure of accuracy. In no scientific context does "precision" have a sign - it's always a positive measure of how precisely something is known. The sign of a constant's value is unrelated to its precision.

3. **Inconsistent with Documentation Expectations**: The docstring example shows `constants.precision('proton mass')` returning `5.1e-37`, a positive value. This sets the expectation that precision values are non-negative. The documentation doesn't warn users that negative constants will produce negative precision values, leading to unexpected behavior.

4. **Breaks Downstream Calculations**: Users employing precision values in uncertainty propagation, error analysis, or statistical calculations expect non-negative values. Negative precision values can cause incorrect results in formulas that assume standard metrological definitions.

## Relevant Context

The bug affects 33 out of 355 physical constants in scipy.constants - specifically all constants with negative values. The affected constants include important thermodynamic quantities like the Sackur-Tetrode constant and various magnetic moment ratios.

The current implementation at `/home/npc/.local/lib/python3.13/site-packages/scipy/constants/_codata.py:2199` directly divides uncertainty by value without taking the absolute value:
```python
return physical_constants[key][2] / physical_constants[key][0]
```

This is a straightforward oversight where the implementation doesn't account for the standard metrological definition of relative uncertainty.

Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.constants.precision.html
Source code: scipy/constants/_codata.py

## Proposed Fix

```diff
--- a/scipy/constants/_codata.py
+++ b/scipy/constants/_codata.py
@@ -2196,7 +2196,7 @@ def precision(key: str) -> float:

     """
     _check_obsolete(key)
-    return physical_constants[key][2] / physical_constants[key][0]
+    return abs(physical_constants[key][2] / physical_constants[key][0])


 def find(sub: str | None = None, disp: bool = False) -> Any:
```