# Bug Report: pd.cut Silent Failure with Denormal Floats

**Target**: `pandas.cut`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`pd.cut` silently returns all NaN values when binning data containing denormal floating-point numbers, even though all input values are valid and within the expected range.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st, assume
import pandas as pd


@given(
    values=st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=2, max_size=50),
    bins=st.integers(min_value=2, max_value=10)
)
@settings(max_examples=200)
def test_cut_assigns_all_values(values, bins):
    assume(len(set(values)) > 1)
    result = pd.cut(values, bins=bins)
    assert len(result) == len(values)
    non_null = result.notna().sum()
    assert non_null == len(values)
```

**Failing input**: `values=[0.0, 2.2250738585e-313], bins=2`

## Reproducing the Bug

```python
import pandas as pd

values = [0.0, 2.2250738585e-313]
result = pd.cut(values, bins=2)

print("Input values:", values)
print("Result:", result)
print("Expected: 2 non-null categorized values")
print("Actual non-null count:", result.notna().sum())
print("Categories:", result.categories)

assert result.notna().sum() == len(values), \
    f"Expected all {len(values)} values to be binned, but got {result.notna().sum()} non-null values"
```

Output:
```
Input values: [0.0, 2.2250738585e-313]
Result: [NaN, NaN]
Categories (0, interval[float64, right]): []
Expected: 2 non-null categorized values
Actual non-null count: 0
AssertionError: Expected all 2 values to be binned, but got 0 non-null values
```

## Why This Is A Bug

The `pd.cut` function is supposed to bin all values into discrete intervals. The docstring states:

> "Any NA values will be NA in the result. Out of bounds values will be NA in the resulting Series or Categorical object."

However, in this case:
1. The input contains NO NA values (both 0.0 and 2.2250738585e-313 are valid floats)
2. Both values are within the range [min, max] of the data
3. Yet the function returns all NaN values and creates 0 categories

This is silent data loss - the function appears to succeed but produces meaningless output. The value 2.2250738585e-313 is a denormal (subnormal) float, which is a valid IEEE 754 floating-point number. The issue appears to be that when computing bin edges with very small ranges, numerical precision errors cause the binning algorithm to fail silently.

## Fix

The root cause appears to be in the bin edge calculation when the range is very small. When `pd.cut` extends the range by 0.1% (as documented), arithmetic with denormal floats can produce NaN or lose precision. The fix should:

1. Detect when the input range is below a numerical precision threshold
2. Either raise a clear error or handle denormal floats more carefully in the binning algorithm
3. Add validation to ensure bins are created successfully before proceeding

A defensive fix would add a check after bin edge calculation:

```diff
--- a/pandas/core/reshape/tile.py
+++ b/pandas/core/reshape/tile.py
@@ -480,6 +480,10 @@ def _bins_to_cuts(
         bins = np.linspace(mx - adj, mn + adj, bins + 1, endpoint=True)
     else:
         bins = np.linspace(mn - adj, mx + adj, bins + 1, endpoint=True)
+
+    if len(bins) < 2 or np.isnan(bins).any():
+        raise ValueError(
+            f"Unable to create valid bins for range [{mn}, {mx}]. "
+            "This may be due to numerical precision issues with very small values.")

     labels = _format_labels(
         bins, precision, right=right, include_lowest=include_lowest
```