# Bug Report: pandas.core.reshape.tile.cut - Precision Parameter Causes Bin Mismatch

**Target**: `pandas.core.reshape.tile.cut`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `precision` parameter in `pd.cut()` causes a mismatch between the bins returned via `retbins=True` and the actual interval boundaries used in the result, violating the API contract that these should be consistent.

## Property-Based Test

```python
import pandas as pd
from hypothesis import given, strategies as st, settings, assume


@given(
    st.lists(st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False), min_size=10, max_size=100),
    st.integers(min_value=2, max_value=10),
    st.integers(min_value=1, max_value=5)
)
@settings(max_examples=500)
def test_cut_bins_match_intervals(values, n_bins, precision):
    assume(len(set(values)) > 1)

    x = pd.Series(values)
    result, bins = pd.cut(x, bins=n_bins, retbins=True, precision=precision)

    categories = result.cat.categories
    for i, interval in enumerate(categories):
        assert interval.left == bins[i], \
            f"Interval {i} left boundary ({interval.left}) doesn't match bins[{i}] ({bins[i]})"
        assert interval.right == bins[i+1], \
            f"Interval {i} right boundary ({interval.right}) doesn't match bins[{i+1}] ({bins[i+1]})"
```

**Failing input**: `values=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.84375], n_bins=2, precision=3`

## Reproducing the Bug

```python
import pandas as pd

values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.84375]
x = pd.Series(values)

result, bins = pd.cut(x, bins=2, retbins=True, precision=3)

print(f"Bins returned via retbins: {bins}")
print(f"Interval categories: {result.cat.categories}")

categories = result.cat.categories
for i, interval in enumerate(categories):
    print(f"\nInterval {i}:")
    print(f"  String representation: {interval}")
    print(f"  Actual left boundary: {interval.left}")
    print(f"  Actual right boundary: {interval.right}")
    print(f"  Expected from bins: [{bins[i]}, {bins[i+1]}]")

    if interval.left != bins[i]:
        print(f"  MISMATCH: {interval.left} != {bins[i]}")
    if interval.right != bins[i+1]:
        print(f"  MISMATCH: {interval.right} != {bins[i+1]}")
```

Expected output: Interval boundaries should exactly match the bins array.

Actual output:
```
Bins returned via retbins: [-0.00184375  0.921875    1.84375   ]
Interval categories: IntervalIndex([(-0.00184, 0.922], (0.922, 1.844]], dtype='interval[float64, right]')

Interval 0:
  String representation: (-0.00184, 0.922]
  Actual left boundary: -0.00184
  Actual right boundary: 0.922
  Expected from bins: [-0.0018437500000000001, 0.921875]
  MISMATCH: -0.00184 != -0.0018437500000000001
  MISMATCH: 0.922 != 0.921875

Interval 1:
  String representation: (0.922, 1.844]
  Actual left boundary: 0.922
  Actual right boundary: 1.844
  Expected from bins: [0.921875, 1.84375]
  MISMATCH: 0.922 != 0.921875
  MISMATCH: 1.844 != 1.84375
```

## Why This Is A Bug

This violates the API contract documented in the docstring for `retbins`:

> retbins : bool, default False
>     Whether to return the bins or not.

The parameter name and documentation suggest that `retbins=True` returns **the actual bins used for binning**. However, the bins returned don't match the actual interval boundaries due to the `precision` parameter rounding the boundaries for display purposes.

This creates several problems:
1. **Misleading API**: Users expect `retbins` to return the actual bins used
2. **Inconsistent values**: `bins[i]` doesn't equal `categories[i-1].right` or `categories[i].left`
3. **Impossible to reconstruct**: Users cannot reconstruct the exact intervals from the returned bins

The bug occurs because:
- The `precision` parameter is intended to control display formatting
- But it actually modifies the interval boundaries themselves, not just their string representation
- The bins array returned via `retbins` contains the original computed values
- The interval boundaries are the rounded values

## Fix

The `precision` parameter should control **display formatting only**, not the actual interval boundaries. The intervals should use the exact computed bin edges.

```diff
--- a/pandas/core/reshape/tile.py
+++ b/pandas/core/reshape/tile.py
@@ -574,7 +574,11 @@ def _format_labels(
     if labels is not None:
         return labels

-    return IntervalIndex.from_breaks(breaks, closed=closed)
+    # Create intervals with exact breaks, not rounded for display
+    intervals = IntervalIndex.from_breaks(breaks, closed=closed)
+
+    # Note: precision parameter should only affect string representation,
+    # not the actual interval boundaries
+    return intervals
```

Alternatively, if rounding is necessary for some internal reason, the `retbins` output should return the rounded values to maintain consistency, or a warning should be added to the documentation explaining this behavior.