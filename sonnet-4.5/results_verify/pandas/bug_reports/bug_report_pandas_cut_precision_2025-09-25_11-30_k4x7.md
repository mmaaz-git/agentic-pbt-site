# Bug Report: pandas.core.reshape.tile.cut Precision Handling

**Target**: `pandas.core.reshape.tile.cut`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`pd.cut()` silently converts all valid numeric data to NaN when the data range is extremely small (< 1e-300), due to a precision handling bug in the `_round_frac` function that produces NaN bin edges.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
import pandas as pd

@given(
    data=st.lists(
        st.floats(
            min_value=-1e6,
            max_value=1e6,
            allow_nan=False,
            allow_infinity=False,
        ),
        min_size=10,
        max_size=100,
    ),
    bins=st.integers(min_value=2, max_value=10),
)
def test_cut_all_values_assigned_to_bins(data, bins):
    assume(len(set(data)) > 1)

    result = pd.cut(data, bins=bins)

    non_na_input = sum(1 for x in data if not pd.isna(x))
    non_na_result = sum(1 for x in result if not pd.isna(x))

    assert non_na_input == non_na_result
```

**Failing input**: `data=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.225073858507e-311], bins=2`

## Reproducing the Bug

```python
import pandas as pd
import numpy as np

data = [0.0] * 9 + [2.225073858507e-311]
result, bins = pd.cut(data, bins=2, retbins=True)

print(f"Input: 10 valid numeric values")
print(f"Bins: {bins}")
print(f"Result categories: {result.categories}")
print(f"Non-NA in result: {sum(1 for x in result if not pd.isna(x))}")
```

Expected: All 10 values should be assigned to bins.
Actual: All values become NaN, with 0 categories created.

## Why This Is A Bug

When data has an extremely small range (e.g., values near 1e-311), the internal `_round_frac` function calculates an excessively large `digits` parameter for `np.around()`, causing it to return NaN for all bin edges. This creates an IntervalIndex with NaN intervals that cannot match any input values, violating the fundamental invariant that `pd.cut()` should bin all valid input data.

The root cause is in `/pandas/core/reshape/tile.py:_round_frac()` at line 624:
```python
digits = -int(np.floor(np.log10(abs(frac)))) - 1 + precision
```

For `frac = 2.225e-311`, this calculates `digits ≈ 316`, which causes `np.around(x, 316)` to overflow and return NaN.

## Fix

```diff
--- a/pandas/core/reshape/tile.py
+++ b/pandas/core/reshape/tile.py
@@ -615,6 +615,9 @@ def _round_frac(x, precision: int):
 def _round_frac(x, precision: int):
     """
     Round the fractional part of the given number
     """
     if not np.isfinite(x) or x == 0:
         return x
     else:
         frac, whole = np.modf(x)
         if whole == 0:
             digits = -int(np.floor(np.log10(abs(frac)))) - 1 + precision
+            # Clamp digits to prevent np.around overflow
+            digits = min(digits, 15)  # float64 has ~15-17 decimal digits of precision
         else:
             digits = precision
         return np.around(x, digits)
```

Alternatively, a more robust fix would be to avoid the precision rounding entirely when bin edges are already sufficiently precise, or to handle the NaN case in `_format_labels` by falling back to the original bins.