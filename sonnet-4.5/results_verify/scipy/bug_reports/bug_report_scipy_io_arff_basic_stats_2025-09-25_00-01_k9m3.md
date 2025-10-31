# Bug Report: scipy.io.arff NumericAttribute._basic_stats Division by Zero

**Target**: `scipy.io.arff._arffread.NumericAttribute._basic_stats`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `NumericAttribute._basic_stats` method raises `ZeroDivisionError` when called with an array containing a single element due to division by `(data.size - 1)` on line 227.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from scipy.io.arff._arffread import NumericAttribute
import numpy as np


@given(st.floats(allow_nan=False, allow_infinity=False))
def test_basic_stats_single_element(value):
    attr = NumericAttribute('test')
    data = np.array([value])
    min_val, max_val, mean_val, std_val = attr._basic_stats(data)
```

**Failing input**: Any single-element array, e.g., `np.array([5.0])`

## Reproducing the Bug

```python
from scipy.io.arff._arffread import NumericAttribute
import numpy as np

attr = NumericAttribute('test')
data = np.array([5.0])
min_val, max_val, mean_val, std_val = attr._basic_stats(data)
```

Output:
```
ZeroDivisionError: float division by zero
```

## Why This Is A Bug

The method computes `nbfac = data.size * 1. / (data.size - 1)` on line 227. When `data.size` is 1, this evaluates to `1.0 / 0`, causing a `ZeroDivisionError`. While this method appears to be unused in the current codebase (dead code), it's still a bug because it's a public API method (single underscore prefix) that crashes on valid input.

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