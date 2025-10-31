# Bug Report: pandas.cut() Loses Values with duplicates='drop'

**Target**: `pandas.core.reshape.tile.cut`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When `pd.cut()` is called with `duplicates='drop'` on data with a very small range, it incorrectly marks most valid input values as NaN. The bins after dropping duplicates don't cover all the input values.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume, settings
import pandas as pd
import numpy as np


@given(
    x=st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=0, max_value=100), min_size=10, max_size=50),
    bins=st.integers(min_value=2, max_value=5)
)
@settings(max_examples=200)
def test_cut_with_labels_false_returns_integers(x, bins):
    assume(len(set(x)) > 1)
    result = pd.cut(x, bins, labels=False, duplicates='drop')
    assert isinstance(result, np.ndarray)
    non_na_input = len([v for v in x if not np.isnan(v)])
    non_na_output = np.sum(~np.isnan(result))
    assert non_na_input == non_na_output
```

**Failing input**: `x=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5e-324], bins=2`

## Reproducing the Bug

```python
import pandas as pd
import numpy as np

x = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5e-324]
result, bins = pd.cut(x, bins=2, labels=False, duplicates='drop', retbins=True)

print("Input:", x)
print("Bins:", bins)
print("Result:", result)
print("Non-NA count:", np.sum(~np.isnan(result)))
```

**Output**:
```
Input: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5e-324]
Bins: [0.e+000 5.e-324]
Result: [nan nan nan nan nan nan nan nan nan  0.]
Non-NA count: 1
```

## Why This Is A Bug

When duplicate bin edges are dropped, the resulting bins `[0.0, 5e-324]` create a single interval `(0.0, 5e-324]`. Since intervals are right-closed by default, all the `0.0` values fall on the left boundary and are excluded, becoming NaN. This violates the fundamental expectation that all non-NaN input values should be binned.

The issue occurs in `_bins_to_cuts()` (tile.py, lines 440-447). After dropping duplicate bins, the function doesn't adjust the interval boundaries to ensure all input values are covered. Specifically, when bins are dropped, `include_lowest` should be automatically set to `True` for the remaining interval, or the left boundary should be adjusted.

## Fix

The fix should ensure that when duplicate bins are dropped and only edge bins remain, the resulting intervals still cover all input values. One approach:

```diff
diff --git a/pandas/core/reshape/tile.py b/pandas/core/reshape/tile.py
index abc123..def456 100644
--- a/pandas/core/reshape/tile.py
+++ b/pandas/core/reshape/tile.py
@@ -444,6 +444,10 @@ def _bins_to_cuts(
                 f"You can drop duplicate edges by setting the 'duplicates' kwarg"
             )
         bins = unique_bins
+        # When bins are reduced, ensure we include the lowest value
+        # to avoid losing values at the minimum
+        if len(bins) == 2:
+            include_lowest = True

     side: Literal["left", "right"] = "left" if right else "right"
```

This ensures that when duplicates are dropped and we're left with just 2 bin edges (one interval), the lowest value is included in the interval.