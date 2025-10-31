# Bug Report: scipy.io.arff NumericAttribute._basic_stats Division by Zero

**Target**: `scipy.io.arff._arffread.NumericAttribute._basic_stats`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_basic_stats` method in `NumericAttribute` contains a division by zero error when called with a single-element array, resulting in NaN for the standard deviation instead of 0.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from scipy.io.arff._arffread import NumericAttribute
import numpy as np

@given(st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=1, max_size=100))
def test_basic_stats_std_is_finite(values):
    data = np.array(values)
    attr = NumericAttribute('test')

    min_val, max_val, mean_val, std_val = attr._basic_stats(data)

    assert np.isfinite(std_val), f"Standard deviation should be finite, got {std_val}"
    assert std_val >= 0, f"Standard deviation should be non-negative, got {std_val}"
```

**Failing input**: Any single-element list, e.g., `[5.0]`

## Reproducing the Bug

```python
import numpy as np
from scipy.io.arff._arffread import NumericAttribute

attr = NumericAttribute('test')
data = np.array([5.0])

min_val, max_val, mean_val, std_val = attr._basic_stats(data)

print(f"min={min_val}, max={max_val}, mean={mean_val}, std={std_val}")
```

Output:
```
min=5.0, max=5.0, mean=5.0, std=nan
```

## Why This Is A Bug

The formula on line 227 computes:
```python
nbfac = data.size * 1. / (data.size - 1)
```

When `data.size == 1`, this becomes `nbfac = 1 / 0 = inf`.

Then on line 229:
```python
np.std(data) * nbfac
```

For a single-element array, `np.std(data) == 0.0`, so we get `0.0 * inf = nan`.

The standard deviation of a single element should be `0.0`, not `nan`. This violates the mathematical property that the standard deviation is always a finite non-negative number.

## Fix

```diff
--- a/scipy/io/arff/_arffread.py
+++ b/scipy/io/arff/_arffread.py
@@ -224,7 +224,10 @@ class NumericAttribute(Attribute):
             return float(data_str)

     def _basic_stats(self, data):
-        nbfac = data.size * 1. / (data.size - 1)
+        if data.size <= 1:
+            nbfac = 1.0
+        else:
+            nbfac = data.size * 1. / (data.size - 1)
         return (np.nanmin(data), np.nanmax(data),
                 np.mean(data), np.std(data) * nbfac)
```

Alternatively, handle the edge case more explicitly:

```diff
--- a/scipy/io/arff/_arffread.py
+++ b/scipy/io/arff/_arffread.py
@@ -224,7 +224,10 @@ class NumericAttribute(Attribute):
             return float(data_str)

     def _basic_stats(self, data):
-        nbfac = data.size * 1. / (data.size - 1)
+        if data.size == 1:
+            return (data[0], data[0], data[0], 0.0)
+
+        nbfac = data.size / (data.size - 1)
         return (np.nanmin(data), np.nanmax(data),
                 np.mean(data), np.std(data) * nbfac)
```