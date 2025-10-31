# Bug Report: pandas.core.reshape.tile qcut() Uneven Quartile Distribution

**Target**: `pandas.core.reshape.tile.qcut`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `qcut()` function with `q=4` and `duplicates='drop'` produces significantly uneven quartiles when the input data contains duplicate values. For example, with 5 values including duplicates, it creates quartiles of sizes [3, 1, 1], where the difference between largest and smallest quartiles is 2, violating the expected equal-sized distribution property.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
import pandas as pd

@given(st.lists(st.floats(0, 100, allow_nan=False), min_size=4, max_size=50))
def test_qcut_quartile_property(data):
    assume(len(set(data)) >= 4)

    result = pd.qcut(data, q=4, duplicates='drop')
    value_counts = result.value_counts()

    if len(value_counts) > 1:
        min_count = value_counts.min()
        max_count = value_counts.max()
        assert max_count - min_count <= 1, f"Quartile sizes too uneven: {value_counts.tolist()}"
```

**Failing input**: `data=[0.0, 0.0, 1.0, 2.0, 3.0]`

## Reproducing the Bug

```python
import pandas as pd

data = [0.0, 0.0, 1.0, 2.0, 3.0]
result = pd.qcut(data, q=4, duplicates='drop')
value_counts = result.value_counts()
print(f"Quartile counts: {value_counts.tolist()}")
```

Output:
```
Quartile counts: [3, 1, 1]
```

The expected behavior for quartiles is that each should contain approximately the same number of elements (differing by at most 1). However, we get one quartile with 3 elements and two with only 1 element each.

## Why This Is A Bug

While this might be considered expected behavior when `duplicates='drop'` merges bins with duplicate edges, the result violates the fundamental property of quantile-based discretization: creating equal-sized buckets. The documentation states that qcut should "Discretize variable into equal-sized buckets based on rank or based on sample quantiles."

When duplicate bin edges are dropped, the remaining bins should still attempt to maintain relatively equal sizes, or at minimum, the documentation should clarify that `duplicates='drop'` can produce highly uneven distributions.

## Fix

This is a design issue rather than a simple code fix. The behavior should either be:

1. **Option A**: Improve the algorithm to redistribute values more evenly after dropping duplicate bins
2. **Option B**: Document this limitation clearly in the docstring, warning users that `duplicates='drop'` may produce uneven bucket sizes
3. **Option C**: Raise a warning when the resulting distribution is highly uneven

Recommended fix is Option B (documentation):

```diff
--- a/pandas/core/reshape/tile.py
+++ b/pandas/core/reshape/tile.py
@@ -300,6 +300,10 @@ def qcut(
         Whether to return the (bins, labels) or not. Can be useful if bins
         is given as a scalar.
     duplicates : {default 'raise', 'drop'}, optional
         If bin edges are not unique, raise ValueError or drop non-uniques.
+        Note: When 'drop' is used, the resulting buckets may have significantly
+        uneven sizes, as bins with duplicate edges are merged. This can violate
+        the equal-sized bucket property, especially with data containing many
+        duplicate values.
```