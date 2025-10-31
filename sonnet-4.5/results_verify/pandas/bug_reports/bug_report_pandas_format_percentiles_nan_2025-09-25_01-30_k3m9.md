# Bug Report: pandas.core.methods format_percentiles produces 'nan%' labels

**Target**: `pandas.core.methods.describe.format_percentiles`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `format_percentiles` function in pandas.core.methods.describe produces invalid 'nan%' labels when given percentiles with very small differences, causing corrupted output in `Series.describe()` and `DataFrame.describe()`.

## Property-Based Test

```python
from pandas.core.methods.describe import format_percentiles
from hypothesis import given, strategies as st


@given(percentiles=st.lists(
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    min_size=2, max_size=10
))
def test_format_percentiles_should_not_produce_nan(percentiles):
    result = format_percentiles(percentiles)

    for formatted in result:
        assert 'nan%' not in formatted.lower(), \
            f"format_percentiles produced 'nan%' for input {percentiles}: {result}"
```

**Failing input**: `[0.0, 2.225073858507e-311]`

## Reproducing the Bug

```python
import pandas as pd
from pandas.core.methods.describe import format_percentiles

percentiles = [0.0, 2.225073858507e-311]
result = format_percentiles(percentiles)
print(result)

s = pd.Series([1, 2, 3, 4, 5])
print(s.describe(percentiles=percentiles))
```

**Output:**
```
['nan%', 'nan%']

count    5.000000
mean     3.000000
std      1.581139
min      1.000000
nan%     1.000000
nan%     1.000000
nan%     3.000000
max      5.000000
dtype: float64
```

## Why This Is A Bug

1. The input percentiles are valid (between 0 and 1)
2. The output 'nan%' is not a valid percentage label
3. This corrupts the index labels in `Series.describe()` and `DataFrame.describe()` output
4. The function doesn't validate or reject such inputs, but produces nonsensical output

The root cause is numerical overflow when computing precision based on extremely small differences between percentiles. The code in `pandas/io/formats/format.py` calculates:
```python
prec = -np.floor(np.log10(np.min(diff))).astype(int)
```

When `diff` is extremely small (approaching machine epsilon), `log10(diff)` becomes a very large negative number, causing overflow when negated and cast to int, resulting in NaN values propagating through the calculation.

## Fix

The function should handle edge cases where percentile differences are extremely small or zero by clamping the precision to a reasonable maximum value:

```diff
--- a/pandas/io/formats/format.py
+++ b/pandas/io/formats/format.py
@@ -1611,7 +1611,10 @@ def format_percentiles(
     else:
         diff = np.diff(percentiles)
         if np.all(diff == 0):
             prec = 0
         else:
-            prec = -np.floor(np.log10(np.min(diff))).astype(int)
+            min_diff = np.min(diff[diff > 0]) if np.any(diff > 0) else 1.0
+            prec = -np.floor(np.log10(min_diff))
+            prec = np.clip(prec, 0, 15).astype(int)

     out = np.empty(len(percentiles), dtype=object)
```

This fix:
1. Filters out zero differences to avoid log10(0)
2. Clamps precision to a reasonable range [0, 15] to prevent overflow
3. Ensures output is always valid percentage strings