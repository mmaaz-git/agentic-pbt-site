# Bug Report: pandas.core.reshape.tile.qcut Numerical Precision Issues and Unequal Bins

**Target**: `pandas.core.reshape.tile.qcut`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `qcut` function has two related issues when dealing with data near float64 precision limits or with many duplicate values:
1. Crashes with ValueError when quantile boundaries result in extremely small differences
2. Produces severely unequal bin sizes despite claiming to create "equal-sized buckets"

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
import pandas as pd
from pandas.core.reshape.tile import qcut

@given(st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=20, max_size=200),
       st.integers(min_value=2, max_value=10))
def test_qcut_equal_sized_bins(data, num_quantiles):
    assume(len(set(data)) >= num_quantiles)

    result, bins = qcut(data, num_quantiles, retbins=True, duplicates='drop')

    unique_bins = result[pd.notna(result)].unique()
    assume(len(unique_bins) >= 2)

    bin_counts = result.value_counts()

    expected_count = len([x for x in data if not pd.isna(x)]) / len(unique_bins)
    tolerance = expected_count * 0.5

    for count in bin_counts:
        assert abs(count - expected_count) <= tolerance
```

**Failing input 1 (crash)**: `data=[0.0]*19 + [2.2250738585e-313], num_quantiles=2`

**Failing input 2 (unequal bins)**: `data=[0.0]*18 + [1.0, -1.0], num_quantiles=2`

## Reproducing the Bug

```python
import pandas as pd
from pandas.core.reshape.tile import qcut

print("Bug 1: Crash with tiny quantile differences")
data_crash = [0.0] * 19 + [2.2250738585e-313]
try:
    result, bins = qcut(data_crash, 2, retbins=True, duplicates='drop')
except ValueError as e:
    print(f"ERROR: {e}")

print("\nBug 2: Severely unequal bin sizes")
data_unequal = [0.0] * 18 + [1.0, -1.0]
result, bins = qcut(data_unequal, 2, retbins=True, duplicates='drop')
print(f"Bin counts:\n{result.value_counts()}")
print(f"Expected: ~10 per bin, Got: 19 and 1")
```

## Why This Is A Bug

The `qcut` function documentation explicitly states it should "Discretize variable into equal-sized buckets based on rank or based on sample quantiles." This is a core promise of the function.

**Bug 1**: When quantile boundaries are computed for data with extremely small differences, the resulting bin edges cause the same numerical precision issues as in `cut`, leading to a crash. Valid inputs should not crash.

**Bug 2**: When data has many duplicates, `qcut` produces severely unequal bins (e.g., 19 values in one bin, 1 in another, when requesting 2 quantiles). This violates the documented behavior of creating "equal-sized buckets". While exact equality is impossible with duplicates, a 19:1 ratio when expecting 10:10 is excessive.

## Fix

The fix requires addressing two issues:

1. **Numerical precision**: Use the same fix as proposed for `cut` to handle extremely small quantile differences
2. **Duplicate handling**: Improve the quantile boundary selection when duplicates are present to better balance bin sizes

```diff
--- a/pandas/core/reshape/tile.py
+++ b/pandas/core/reshape/tile.py
@@ -320,6 +320,16 @@ def qcut(
     else:
         quantiles = np.linspace(0, 1, q + 1)
     bins = algos.quantile(x_idx, quantiles)
+
+    # Handle numerical precision issues when quantiles are very close
+    bin_diffs = np.diff(bins)
+    if np.any((bin_diffs > 0) & (bin_diffs < 1e-300)):
+        # Adjust bins to avoid precision issues
+        for i in range(len(bins) - 1):
+            if 0 < bin_diffs[i] < 1e-300:
+                bins[i+1] = bins[i] + 1e-299
+
     fac, bins = _bins_to_cuts(
         x_idx,
         bins,
```

Note: This is a partial fix addressing the numerical precision crash. The unequal bins issue with duplicates is more complex and may require rethinking the quantile selection strategy when `duplicates='drop'` is used.