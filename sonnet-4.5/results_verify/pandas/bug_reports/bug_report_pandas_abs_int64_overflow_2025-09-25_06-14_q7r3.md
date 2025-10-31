# Bug Report: pandas Series.abs() integer overflow on min int64

**Target**: `pandas.Series.abs`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`Series.abs()` returns a negative value for the minimum int64 value (-9223372036854775808), violating the mathematical property that absolute value is always non-negative.

## Property-Based Test

```python
from hypothesis import given, settings
from hypothesis.extra.pandas import series
import pandas as pd

@given(series(dtype=int))
@settings(max_examples=200)
def test_series_abs_non_negative(s):
    result = s.abs()
    assert (result >= 0).all()
```

**Failing input**: `Series([-9223372036854775808])`

## Reproducing the Bug

```python
import pandas as pd
import numpy as np

min_int64 = np.iinfo(np.int64).min
s = pd.Series([min_int64])

print(f"Input: {s.values[0]}")
result = s.abs()
print(f"abs() result: {result.values[0]}")
print(f"Is non-negative: {result.values[0] >= 0}")
```

Output:
```
Input: -9223372036854775808
abs() result: -9223372036854775808
Is non-negative: False
```

## Why This Is A Bug

The absolute value function has a fundamental mathematical property: abs(x) ≥ 0 for all x. The pandas documentation states that `abs()` returns "the absolute value of each element" without documenting this overflow behavior.

This silent failure can lead to incorrect computations in downstream code that assumes abs() always returns non-negative values (e.g., distance calculations, magnitude computations).

While this behavior matches numpy's `np.abs()`, pandas has the opportunity to provide better semantics by detecting overflow and either:
1. Promoting to float64 or a larger integer type
2. Raising an error/warning
3. Documenting this limitation explicitly

## Fix

Detect when abs() would overflow and handle it appropriately:

```diff
def abs(self):
-   return self._unary_method(np.abs)
+   result = self._unary_method(np.abs)
+   # Check for overflow: if result is negative, input was min_int
+   mask = (result < 0) & (self.dtype.kind == 'i')
+   if mask.any():
+       # Promote to float to avoid overflow
+       return self.astype(float).abs()
+   return result
```

Alternatively, document this limitation in the docstring with a warning.