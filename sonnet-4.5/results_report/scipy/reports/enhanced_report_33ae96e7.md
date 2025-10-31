# Bug Report: scipy.constants.precision() Returns Negative Values for Negative Physical Constants

**Target**: `scipy.constants.precision()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `precision()` function returns negative relative precision values when physical constants have negative values, violating the standard physics definition where relative precision must always be non-negative.

## Property-Based Test

```python
import math
from hypothesis import given, settings, strategies as st
import scipy.constants as sc
import pytest

all_keys = list(sc.physical_constants.keys())

@given(st.sampled_from(all_keys))
@settings(max_examples=500)
def test_precision_calculation(key):
    result = sc.precision(key)
    value_const, unit_const, abs_precision = sc.physical_constants[key]

    if value_const == 0:
        pytest.skip("Cannot compute relative precision for zero value")

    expected = abs(abs_precision / value_const)
    assert math.isclose(result, expected, rel_tol=1e-9, abs_tol=1e-15)

if __name__ == "__main__":
    # Run the test
    test_precision_calculation()
```

<details>

<summary>
**Failing input**: `key='electron-proton magn. moment ratio'`
</summary>
```
/home/npc/pbt/agentic-pbt/worker_/58/hypo.py:11: ConstantWarning: Constant 'Wien displacement law constant' is not in current CODATA 2022 data set
  result = sc.precision(key)
/home/npc/pbt/agentic-pbt/worker_/58/hypo.py:11: ConstantWarning: Constant 'atomic unit of 1st hyperpolarizablity' is not in current CODATA 2022 data set
  result = sc.precision(key)
/home/npc/pbt/agentic-pbt/worker_/58/hypo.py:11: ConstantWarning: Constant 'electron to shielded helion magn. moment ratio' is not in current CODATA 2022 data set
  result = sc.precision(key)
/home/npc/pbt/agentic-pbt/worker_/58/hypo.py:11: ConstantWarning: Constant 'electron-proton magn. moment ratio' is not in current CODATA 2022 data set
  result = sc.precision(key)
/home/npc/pbt/agentic-pbt/worker_/58/hypo.py:11: ConstantWarning: Constant 'muon magn. moment to Bohr magneton ratio' is not in current CODATA 2022 data set
  result = sc.precision(key)
/home/npc/pbt/agentic-pbt/worker_/58/hypo.py:11: ConstantWarning: Constant 'deuteron magn. moment to Bohr magneton ratio' is not in current CODATA 2022 data set
  result = sc.precision(key)
/home/npc/pbt/agentic-pbt/worker_/58/hypo.py:11: ConstantWarning: Constant 'deuteron-proton magn. moment ratio' is not in current CODATA 2022 data set
  result = sc.precision(key)
/home/npc/pbt/agentic-pbt/worker_/58/hypo.py:11: ConstantWarning: Constant 'electron-muon magn. moment ratio' is not in current CODATA 2022 data set
  result = sc.precision(key)
/home/npc/pbt/agentic-pbt/worker_/58/hypo.py:11: ConstantWarning: Constant 'electron-neutron magn. moment ratio' is not in current CODATA 2022 data set
  result = sc.precision(key)
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/58/hypo.py", line 22, in <module>
    test_precision_calculation()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/58/hypo.py", line 9, in test_precision_calculation
    @settings(max_examples=500)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/58/hypo.py", line 18, in test_precision_calculation
    assert math.isclose(result, expected, rel_tol=1e-9, abs_tol=1e-15)
           ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_precision_calculation(
    key='electron-proton magn. moment ratio',
)
```
</details>

## Reproducing the Bug

```python
import scipy.constants as sc

# Test the precision function with a negative-valued physical constant
key = 'electron magn. moment'
result = sc.precision(key)
value_const, unit_const, abs_precision = sc.physical_constants[key]

print(f"Physical constant: {key}")
print(f"Value: {value_const}")
print(f"Unit: {unit_const}")
print(f"Absolute precision: {abs_precision}")
print(f"precision(key) returned: {result}")
print(f"Expected (using standard physics definition): {abs(abs_precision / value_const)}")
print()

# The standard physics definition of relative precision should always be positive
# as it represents the magnitude of uncertainty relative to the measured value
assert result > 0, f"Relative precision should be positive, but got {result}"
```

<details>

<summary>
AssertionError: Relative precision should be positive, but got -8.61626627947119e-08
</summary>
```
/home/npc/pbt/agentic-pbt/worker_/58/repo.py:5: ConstantWarning: Constant 'electron magn. moment' is not in current CODATA 2022 data set
  result = sc.precision(key)
Physical constant: electron magn. moment
Value: -9.28476412e-24
Unit: J T^-1
Absolute precision: 8e-31
precision(key) returned: -8.61626627947119e-08
Expected (using standard physics definition): 8.61626627947119e-08

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/58/repo.py", line 18, in <module>
    assert result > 0, f"Relative precision should be positive, but got {result}"
           ^^^^^^^^^^
AssertionError: Relative precision should be positive, but got -8.61626627947119e-08
```
</details>

## Why This Is A Bug

Relative precision (also known as relative uncertainty) is a fundamental concept in physics and measurement science that represents the ratio of absolute uncertainty to the magnitude of the measured value. By definition, it quantifies the size of uncertainty and must always be non-negative. The current implementation violates this standard physics definition by returning negative values for 58 physical constants (13% of all constants in scipy.constants).

The scipy documentation states the function returns "relative precision" without explicitly specifying sign behavior. However, the term "relative precision" has an unambiguous meaning in physics where it's always |uncertainty/value| or uncertainty/|value|. The documentation example shows `constants.precision('proton mass')` returning `5.1e-37` (positive), implying positive values are expected.

This affects all physical constants with negative values, primarily magnetic moments and related quantities. A negative relative precision is physically meaningless - uncertainty cannot be negative as it represents the magnitude of measurement error, not a signed quantity.

## Relevant Context

The bug affects 58 out of 445 physical constants in scipy.constants, including:
- All magnetic moment constants (electron, muon, neutron, proton, etc.)
- Magnetic moment ratios
- Electron charge to mass quotient
- Sackur-Tetrode constants

The current implementation in scipy/constants/_codata.py directly divides the absolute precision by the value:
```python
return physical_constants[key][2] / physical_constants[key][0]
```

When the constant's value is negative (physical_constants[key][0] < 0) and the absolute precision is positive (physical_constants[key][2] > 0), this produces a negative result, contradicting standard physics practice.

Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.constants.precision.html
Source code location: scipy/constants/_codata.py (function `precision`)

## Proposed Fix

```diff
--- a/scipy/constants/_codata.py
+++ b/scipy/constants/_codata.py
@@ -XX,7 +XX,7 @@ def precision(key: str) -> float:
     """
     _check_obsolete(key)
-    return physical_constants[key][2] / physical_constants[key][0]
+    return abs(physical_constants[key][2] / physical_constants[key][0])
```