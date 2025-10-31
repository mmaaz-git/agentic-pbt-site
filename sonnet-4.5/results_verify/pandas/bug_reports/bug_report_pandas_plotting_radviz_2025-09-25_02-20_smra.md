# Bug Report: pandas.plotting.radviz Produces NaN for Constant Columns

**Target**: `pandas.plotting._matplotlib.radviz`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `radviz` function's internal `normalize` helper produces NaN values when a column contains constant values (all the same), causing incorrect visualization coordinates.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas as pd

@given(st.floats(allow_nan=False, allow_infinity=False))
def test_radviz_normalize_constant_column(constant_value):
    df = pd.DataFrame({
        'constant_col': [constant_value] * 3,
        'varying_col': [1.0, 2.0, 3.0],
        'class': ['a', 'b', 'c']
    })

    def normalize(series):
        a = min(series)
        b = max(series)
        return (series - a) / (b - a)

    normalized = normalize(df['constant_col'])

    assert not normalized.isna().any(), \
        "Normalize should not produce NaN for constant column"
```

**Failing input**: Any constant value (e.g., `constant_value=0.0`)

## Reproducing the Bug

```python
import pandas as pd

df = pd.DataFrame({
    'constant_col': [5.0, 5.0, 5.0],
    'varying_col': [1.0, 2.0, 3.0],
    'class': ['a', 'b', 'c']
})

def normalize(series):
    a = min(series)
    b = max(series)
    return (series - a) / (b - a)

normalized = normalize(df['constant_col'])

print("Original column:", list(df['constant_col']))
print("Normalized column:", list(normalized))
print("Contains NaN:", normalized.isna().any())
```

Output:
```
Original column: [5.0, 5.0, 5.0]
Normalized column: [nan, nan, nan]
Contains NaN: True
```

## Why This Is A Bug

When a column has constant values (min == max), the normalization formula `(series - a) / (b - a)` performs division by zero, producing NaN values. These NaN values then propagate through the radviz calculation, leading to incorrect visualization coordinates.

This is problematic because:
1. Constant columns are valid input - users may have features that don't vary in their dataset
2. The function fails silently without error or warning
3. The resulting visualization is incorrect/meaningless
4. Users won't know their visualization is wrong

## Fix

```diff
--- a/pandas/plotting/_matplotlib/tools.py
+++ b/pandas/plotting/_matplotlib/tools.py
@@ -XXX,7 +XXX,10 @@ def radviz(...):
     def normalize(series):
         a = min(series)
         b = max(series)
+        if a == b:
+            return series * 0.0
         return (series - a) / (b - a)
```

When a column is constant, normalize it to all zeros (or could use all 0.5 to center it). This maintains the mathematical properties needed for radviz while handling the edge case gracefully.