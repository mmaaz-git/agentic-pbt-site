# Bug Report: scipy.constants.precision() Returns Negative Values

**Target**: `scipy.constants.precision()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `precision()` function returns negative relative precision values when the physical constant has a negative value, violating the mathematical definition of relative precision which should always be non-negative.

## Property-Based Test

```python
import math
from hypothesis import given, settings, strategies as st
import scipy.constants as sc

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
```

**Failing input**: `key='electron magn. moment'`

## Reproducing the Bug

```python
import scipy.constants as sc

key = 'electron magn. moment'
result = sc.precision(key)
value_const, unit_const, abs_precision = sc.physical_constants[key]

print(f"Value: {value_const}")
print(f"Absolute precision: {abs_precision}")
print(f"precision(key) returned: {result}")
print(f"Expected: {abs(abs_precision / value_const)}")

assert result > 0, f"Precision should be positive, but got {result}"
```

Output:
```
Value: -9.28476412e-24
Absolute precision: 8e-31
precision(key) returned: -8.61626627947119e-08
Expected: 8.61626627947119e-08
AssertionError: Precision should be positive, but got -8.61626627947119e-08
```

## Why This Is A Bug

Relative precision is defined as the ratio of absolute uncertainty to the magnitude of the measured value. It represents a measure of uncertainty and should always be a non-negative quantity. When a physical constant has a negative value (like the electron magnetic moment), the relative precision should still be positive since it's measuring the uncertainty, not the sign of the value.

The current implementation computes `abs_precision / value` but should compute `abs_precision / abs(value)` or equivalently `abs(abs_precision / value)`.

## Fix

```diff
--- a/scipy/constants/_codata.py
+++ b/scipy/constants/_codata.py
@@ -XX,7 +XX,7 @@ def precision(key: str) -> float:
     """
     try:
-        return physical_constants[key][2] / physical_constants[key][0]
+        return abs(physical_constants[key][2] / physical_constants[key][0])
     except KeyError:
         _raise_key_error(key)
```