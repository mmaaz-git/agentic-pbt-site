# Bug Report: scipy.io.arff NumericAttribute._basic_stats Division by Zero

**Target**: `scipy.io.arff._arffread.NumericAttribute._basic_stats`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`NumericAttribute._basic_stats` produces infinite standard deviation when called on single-element arrays due to division by zero at line 227: `nbfac = data.size * 1. / (data.size - 1)` evaluates to `1.0 / 0 = inf`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from scipy.io.arff._arffread import NumericAttribute
import numpy as np

@given(value=st.floats(allow_nan=False, allow_infinity=False,
                       min_value=-1e6, max_value=1e6))
def test_basic_stats_single_element_should_be_finite(value):
    """_basic_stats should produce finite statistics for single-element arrays"""
    attr = NumericAttribute("test")
    data = np.array([value])

    min_val, max_val, mean_val, std_val = attr._basic_stats(data)

    assert np.isfinite(std_val), \
        f"Standard deviation should be finite for single element, got {std_val}"
```

**Failing input**: Any single-element numpy array, e.g., `np.array([5.0])`

## Reproducing the Bug

```python
from scipy.io.arff._arffread import NumericAttribute
import numpy as np

attr = NumericAttribute("test")
data = np.array([5.0])

min_val, max_val, mean_val, std_val = attr._basic_stats(data)

print(f"min: {min_val}")
print(f"max: {max_val}")
print(f"mean: {mean_val}")
print(f"std: {std_val}")
print(f"std is infinite: {np.isinf(std_val)}")
```

Output:
```
min: 5.0
max: 5.0
mean: 5.0
std: inf
std is infinite: True
```

## Why This Is A Bug

Line 227 in `_arffread.py`:
```python
def _basic_stats(self, data):
    nbfac = data.size * 1. / (data.size - 1)
    return (np.nanmin(data), np.nanmax(data),
            np.mean(data), np.std(data) * nbfac)
```

When `data.size == 1`, the calculation becomes:
- `nbfac = 1.0 / (1 - 1) = 1.0 / 0 = inf`
- `std = np.std([5.0]) * inf = 0 * inf = nan` or `inf`

The intended formula appears to be the sample standard deviation with Bessel's correction:
```
sample_std = sqrt(sum((x - mean)^2) / (n - 1))
```

However, this formula is undefined for n=1 (you can't compute sample variance from a single point). The method should either:
1. Return NaN for n=1 (mathematically correct)
2. Return 0 for n=1 (population std = 0)
3. Raise an error for n=1
4. Use population std without correction

**Note**: This method is not currently called anywhere in the scipy.io.arff module, making this a low-priority bug. However, it should still be fixed to prevent future issues if the method is used.

## Fix

```diff
--- a/scipy/io/arff/_arffread.py
+++ b/scipy/io/arff/_arffread.py
@@ -225,7 +225,11 @@ class NumericAttribute(Attribute):

     def _basic_stats(self, data):
-        nbfac = data.size * 1. / (data.size - 1)
+        if data.size <= 1:
+            # Sample std is undefined for n=1; return NaN
+            return (np.nanmin(data), np.nanmax(data),
+                    np.mean(data), np.nan)
+        nbfac = data.size * 1. / (data.size - 1)
         return (np.nanmin(data), np.nanmax(data),
                 np.mean(data), np.std(data) * nbfac)
```

Alternative fix using population std for single elements:
```diff
--- a/scipy/io/arff/_arffread.py
+++ b/scipy/io/arff/_arffread.py
@@ -225,7 +225,11 @@ class NumericAttribute(Attribute):

     def _basic_stats(self, data):
+        if data.size <= 1:
+            # For single element, use population std (which is 0)
+            return (np.nanmin(data), np.nanmax(data),
+                    np.mean(data), 0.0)
         nbfac = data.size * 1. / (data.size - 1)
         return (np.nanmin(data), np.nanmax(data),
                 np.mean(data), np.std(data) * nbfac)
```